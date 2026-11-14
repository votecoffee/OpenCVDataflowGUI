"""Private bounded native-worker transport; executable selection is operator config."""

import base64
import json
import queue
import subprocess
import threading
import time
import uuid

import numpy as np

MAX_LINE = 24 * 1024 * 1024


def unpack(value):
    dtype = np.dtype(value["dtype"])
    shape = value["shape"]
    if dtype not in (np.dtype("uint8"), np.dtype("float32")) or len(shape) not in (2, 3):
        raise ValueError("Unexpected simulator array format")
    if any(type(n) is not int or not 1 <= n <= 1920 for n in shape):
        raise ValueError("Simulator array dimensions out of bounds")
    size = int(np.prod(shape)) * dtype.itemsize
    if size > 12 * 1024 * 1024:
        raise ValueError("Simulator array exceeds 12 MiB")
    raw = base64.b64decode(value["base64"], validate=True)
    if len(raw) != size:
        raise ValueError("Simulator array byte count differs")
    return np.frombuffer(raw, dtype).reshape(shape).copy()


class SimClient:
    def __init__(self, run_id, cancelled=lambda: False):
        from opencv_dataflow.ros2_client import runtime_environment

        python, env = runtime_environment()
        if env.get("ROS_DOMAIN_ID") != "130" or not env.get("GZ_PARTITION", "").startswith(
            "opencv_bin_pick_"
        ):
            raise ValueError("Select the isolated bin-picking simulation environment")
        env.update(
            ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST",
            RCUTILS_LOGGING_USE_STDOUT="0",
            RMW_IMPLEMENTATION="rmw_fastrtps_cpp",
        )
        env.pop("ROS_STATIC_PEERS", None)
        self.run_id = run_id
        self.cancelled = cancelled
        self.server_id = None
        self.environment = None
        self.responses = queue.Queue(maxsize=2)
        self.stderr = bytearray()
        self.process = subprocess.Popen(
            [
                python,
                "-u",
                "-m",
                "opencv_dataflow_pick_backend.worker",
                "--simulation-only",
                "--ros-args",
                "-p",
                "use_sim_time:=true",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._drain, daemon=True).start()
        try:
            self.info = self.call("hello")
        except BaseException:
            self.close()
            raise

    def _read(self):
        try:
            while True:
                line = self.process.stdout.readline(MAX_LINE + 1)
                if not line:
                    raise EOFError("Native simulation worker exited")
                if len(line) > MAX_LINE:
                    raise ValueError("Native response exceeds transport limit")
                self.responses.put_nowait(json.loads(line))
        except Exception as error:
            try:
                self.responses.put_nowait(error)
            except queue.Full:
                pass

    def _drain(self):
        while True:
            data = self.process.stderr.read1(4096)
            if not data:
                return
            self.stderr.extend(data)
            del self.stderr[:-16384]

    def call(self, operation, **parameters):
        request_id = str(uuid.uuid4())
        item = dict(
            request_id=request_id,
            run_id=self.run_id,
            server_id=self.server_id,
            operation=operation,
            parameters=parameters,
        )
        self.process.stdin.write((json.dumps(item, allow_nan=False) + "\n").encode())
        self.process.stdin.flush()
        end = time.monotonic() + 110
        sent_cancel = False
        while True:
            if (self.cancelled() or time.monotonic() > end) and not sent_cancel:
                self.process.stdin.write(b'{"operation":"cancel"}\n')
                self.process.stdin.flush()
                sent_cancel = True
                end = time.monotonic() + 5
            elif sent_cancel and time.monotonic() > end:
                raise RuntimeError(
                    "Cancellation/timeout outcome unknown; stop and reconcile the simulator"
                )
            try:
                result = self.responses.get(timeout=0.05)
            except queue.Empty:
                continue
            if isinstance(result, Exception):
                detail = self.stderr.decode(errors="replace")[-2000:]
                raise RuntimeError(f"Simulation transport failed: {result}\n{detail}") from result
            if (
                result.get("schema") != "opencv_sim_pick_primitive_v1"
                or result.get("simulation_only") is not True
                or result.get("request_id") != request_id
                or result.get("run_id") != self.run_id
            ):
                raise ValueError("Native response correlation differs")
            if self.server_id is None:
                self.server_id = result["server_id"]
                self.environment = result["environment"]
            if result["server_id"] != self.server_id or result["environment"] != self.environment:
                raise ValueError("Native server/fixture identity changed")
            self.last_correlation = {
                k: result[k] for k in ("request_id", "run_id", "server_id", "environment")
            }
            if sent_cancel:
                raise InterruptedError(
                    "Simulation request cancelled; reconcile effects before a new run"
                )
            if result["status"] != "ok":
                raise RuntimeError(result.get("reason", "Native operation failed"))
            return result["value"]

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=4)
            except (OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                self.process.wait(timeout=4)
