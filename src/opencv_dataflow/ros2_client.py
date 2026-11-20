"""Bounded subprocess bridge to a trusted, separately installed Lyrical runtime."""

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from .ros2_support.wire import read_frame, write_frame


class ROSRemoteError(RuntimeError):
    """A worker rejected a request before returning a result."""

    def __init__(self, message, code=""):
        super().__init__(message)
        self.code = code


class ROSPublishUnknown(RuntimeError):
    """A publication might have reached DDS; never retry it automatically."""


def _default_profile(module_file, prefix):
    local = module_file.parents[2] / ".local" / "ros2-runtime.json"
    if not local.is_file() and module_file.is_relative_to(prefix):
        checkout = prefix.parent
        if (checkout / "src/opencv_dataflow/ros2_client.py").is_file():
            return checkout / ".local" / "ros2-runtime.json"
    return local


def runtime_environment():
    """Local operator configuration, deliberately absent from saved projects."""
    configured = os.environ.get("OPENCV_DATAFLOW_ROS2_PROFILE")
    path = (
        Path(configured).expanduser()
        if configured
        else _default_profile(Path(__file__).resolve(), Path(sys.prefix).resolve())
    )
    profile = {}
    if configured and not path.is_file():
        raise FileNotFoundError("The configured ROS runtime profile does not exist")
    if path.is_file():
        if path.stat().st_size > 32768:
            raise ValueError("ROS runtime profile exceeds 32 KiB")
        profile = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(profile, dict) or profile.keys() - {
            "python",
            "prefix",
            "path_prepend",
            "python_paths",
            "ament_prefixes",
            "ros_distro",
        }:
            raise ValueError("Invalid ROS runtime profile fields")
    python = os.environ.get("OPENCV_DATAFLOW_ROS2_PYTHON", profile.get("python", ""))
    if (
        not isinstance(python, str)
        or not python
        or not Path(python).is_absolute()
        or not Path(python).is_file()
    ):
        raise RuntimeError("Configure a ROS 2 Lyrical Python runtime; see docs/ros2.md")
    env = os.environ.copy()
    for key, variable in (
        ("path_prepend", "PATH"),
        ("python_paths", "PYTHONPATH"),
        ("ament_prefixes", "AMENT_PREFIX_PATH"),
    ):
        values = profile.get(key, [])
        if (
            not isinstance(values, list)
            or len(values) > 64
            or any(
                not isinstance(v, str)
                or not Path(v).is_absolute()
                or not Path(v).is_dir()
                or os.pathsep in v
                for v in values
            )
        ):
            raise ValueError(f"Invalid ROS runtime {key} paths")
        # Python extensions must come from the ROS environment, not the GUI environment.
        env[variable] = os.pathsep.join(values) + (
            os.pathsep + env[variable] if values and env.get(variable) else env.get(variable, "")
        )
    env["ROS_DISTRO"] = profile.get("ros_distro", env.get("ROS_DISTRO", ""))
    env["ROS_VERSION"] = "2"
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PYTHONHOME", None)
    if env["ROS_DISTRO"] != "lyrical":
        raise RuntimeError("The configured ROS runtime must be Lyrical")
    return python, env


class ROSClient:
    """One private endpoint per worker; no Qt, OpenCV or rclpy ABI mixing."""

    def __init__(self, config, startup_timeout, cancelled=lambda: False):
        self.process = None
        self._closed = False
        self._responses = queue.Queue(maxsize=2)
        self._serial = threading.Lock()
        self._number = 0
        self._stderr = bytearray()
        self._threads = []
        python, env = runtime_environment()
        env.update(
            {
                "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
                "ROS_DOMAIN_ID": str(config["domain_id"]),
                "ROS_AUTOMATIC_DISCOVERY_RANGE": config["discovery_range"].upper(),
                "RMW_FASTRTPS_PUBLICATION_MODE": config["publication_mode"].upper(),
                "RCUTILS_LOGGING_USE_STDOUT": "0",
            }
        )
        # Do not inherit static peers that could defeat the explicit discovery scope.
        env.pop("ROS_STATIC_PEERS", None)
        self.process = subprocess.Popen(
            [python, "-u", str(Path(__file__).parent / "ros2_support" / "worker.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self._start_thread(self._read, "ros2-rpc-read")
        self._start_thread(self._drain, "ros2-rpc-stderr")
        try:
            self.info = self.request("init", startup_timeout, cancelled, config=config)
        except BaseException:
            self._terminate()
            raise

    def _start_thread(self, target, name):
        thread = threading.Thread(target=target, name=name, daemon=True)
        self._threads.append(thread)
        thread.start()
        return thread

    def _read(self):
        try:
            while not self._closed:
                response = read_frame(self.process.stdout)
                self._responses.put_nowait(response)
        except Exception as exc:
            try:
                self._responses.put_nowait(exc)
            except queue.Full:
                pass

    def _drain(self):
        try:
            while True:
                chunk = self.process.stderr.read1(4096)
                if not chunk:
                    break
                self._stderr.extend(chunk)
                del self._stderr[:-16384]
        except OSError, ValueError:
            pass

    def request(self, operation, deadline_seconds, cancelled=lambda: False, **arguments):
        with self._serial:
            if self._closed or self.process.poll() is not None:
                raise RuntimeError("ROS worker is closed; reset runtime before reconnecting")
            self._number += 1
            number = self._number
            failures = []

            def send():
                try:
                    write_frame(self.process.stdin, {"id": number, "op": operation, **arguments})
                except Exception as exc:
                    failures.append(exc)

            sender = self._start_thread(send, "ros2-rpc-send")
            end = time.monotonic() + deadline_seconds
            try:
                while True:
                    if cancelled():
                        raise InterruptedError("ROS operation stopped")
                    if failures:
                        raise failures[0]
                    remaining = end - time.monotonic()
                    if remaining <= 0:
                        detail = self._stderr.decode("utf-8", errors="replace")[-2000:].strip()
                        raise TimeoutError(
                            "ROS worker request timed out" + (f": {detail}" if detail else "")
                        )
                    try:
                        response = self._responses.get(timeout=min(0.05, remaining))
                    except queue.Empty:
                        continue
                    if isinstance(response, Exception):
                        detail = self._stderr.decode("utf-8", errors="replace")[-2000:].strip()
                        raise RuntimeError(
                            "ROS worker transport closed" + (f": {detail}" if detail else "")
                        ) from response
                    if (
                        not isinstance(response, dict)
                        or response.get("id") != number
                        or type(response.get("ok")) is not bool
                    ):
                        raise ValueError("Invalid ROS worker response")
                    if not response["ok"]:
                        error = response.get("error", {})
                        message = str(error.get("message", "ROS worker rejected the request"))
                        if error.get("unknown_publish") is True:
                            raise ROSPublishUnknown(message)
                        raise ROSRemoteError(message, str(error.get("type", "")))
                    result = response["result"]
                    if not isinstance(result, dict):
                        raise ValueError("Invalid ROS worker result")
                    if operation == "publish" and (
                        type(result.get("published")) is not bool
                        or type(result.get("attempted")) is not bool
                        or result["published"] != result["attempted"]
                    ):
                        raise ValueError("Invalid ROS publication receipt")
                    return result
            except ROSRemoteError:
                raise  # Validation rejected before publication; no ambiguous write.
            except BaseException as exc:
                self._terminate()
                if operation == "publish":
                    raise ROSPublishUnknown(
                        "ROS publish outcome unknown; inspect the receiver before resetting runtime"
                    ) from exc
                raise
            finally:
                sender.join(timeout=1)
                if not sender.is_alive():
                    self._threads.remove(sender)

    def _terminate(self):
        self._closed = True
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(timeout=1)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            stream.close()

    def close(self):
        if self._closed:
            return
        try:
            self.request("close", 3)
            self.process.wait(timeout=3)
        except Exception:
            pass
        finally:
            self._terminate()
