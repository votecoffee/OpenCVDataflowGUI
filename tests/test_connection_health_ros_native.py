"""Opt-in local Lyrical/Fast DDS observers; no robot, PLC or service calls."""

import os
import queue
import subprocess
import threading
import time
import uuid

import pytest

from opencv_dataflow.connection_health import ROSHealth
from opencv_dataflow.ros2_blocks import ROSBlock
from opencv_dataflow.ros2_client import ROSClient, runtime_environment

pytestmark = pytest.mark.skipif(
    os.environ.get("OPENCV_DATAFLOW_TEST_ROS2") != "1", reason="Opt-in local ROS runtime required"
)


def writer():
    return ROSClient(
        {
            **ROSBlock.defaults,
            "domain_id": 219,
            "discovery_range": "localhost",
            "topic": "/health_test/sample_" + uuid.uuid4().hex,
            "message_type": "std_msgs/msg/String",
            "kind": "data",
            "role": "write",
            "reliability": "best_effort",
        },
        30,
    )


def check(block):
    block._next_check = 0  # Force a fresh probe without sleeping between test assertions.
    result = block.execute({})
    assert result is not None, block.error
    return result.outputs


def test_native_node_service_topic_receipt_staleness_and_worker_failure():
    publisher = writer()
    monitor = ROSHealth(
        parameters={
            "enabled": True,
            "domain_id": 219,
            "timeout_seconds": 1,
            "required_nodes": [publisher.info["node"]],
            "required_services": [
                {
                    "name": publisher.info["node"] + "/get_parameters",
                    "type": "rcl_interfaces/srv/GetParameters",
                    "check_response": True,
                }
            ],
            "topics": [
                {
                    "name": publisher.info["topic"],
                    "type": "std_msgs/msg/String",
                    "max_age_seconds": 0.3,
                }
            ],
        }
    )
    clients = []
    try:
        output = check(monitor)
        clients.append(monitor._probe.client)
        assert output["connected"] and output["warning"]  # No receipt yet.
        rows = output["report"]["resources"]
        assert rows[0]["state"] == rows[1]["state"] == "ok"
        assert rows[1]["service_called"] is True and rows[1]["response_confirmed"] is True
        publisher.request(
            "publish",
            5,
            value={"data": "health"},
            timeout=2,
            require_subscriber=True,
            wait_for_ack=False,
            stamp_mode="preserve",
        )
        assert check(monitor)["ok"]
        time.sleep(0.35)
        assert check(monitor)["warning"]
        monitor._probe.client.process.terminate()
        monitor._probe.client.process.wait(timeout=5)
        assert check(monitor)["alarm"]
        monitor.parameters.update(required_nodes=[], required_services=[], topics=[])
        assert check(monitor)["ok"]  # New observer recovers; no remote checks configured.
        clients.append(monitor._probe.client)
    finally:
        monitor.close()
        publisher.close()
    assert all(c.process.poll() is not None for c in [publisher, *clients])
    assert all(not t.is_alive() for c in [publisher, *clients] for t in c._threads)


def test_native_missing_service_is_alarm_with_runtime_still_connected():
    block = ROSHealth(
        parameters={
            "enabled": True,
            "domain_id": 219,
            "timeout_seconds": 0.15,
            "required_services": [
                {"name": "/missing/service_" + uuid.uuid4().hex, "type": "std_srvs/srv/Trigger"}
            ],
        }
    )
    try:
        output = check(block)
        assert output["connected"] and output["alarm"]
        assert output["report"]["resources"][0]["service_called"] is False
    finally:
        block.close()


def test_native_qos_and_uninstalled_topic_interfaces_are_resource_warnings():
    publisher = writer()
    monitor = ROSHealth(
        parameters={
            "enabled": True,
            "domain_id": 219,
            "timeout_seconds": 0.3,
            "reliability": "reliable",
            "topics": [
                {"name": publisher.info["topic"], "type": "std_msgs/msg/String"},
                {"name": "/health_test/missing_interface", "type": "absent_pkg/msg/Missing"},
            ],
        }
    )
    try:
        output = check(monitor)
        assert output["connected"] and output["warning"]
        rows = output["report"]["resources"]
        assert rows[0]["matched_publishers"] == 0 and rows[0]["state"] == "warning"
        assert rows[1]["error_type"] and rows[1]["state"] == "warning"
    finally:
        monitor.close()
        publisher.close()


def test_native_advertised_but_unresponsive_read_only_service_is_red():
    python, env = runtime_environment()
    env.update(
        ROS_DOMAIN_ID="219",
        ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST",
        RMW_IMPLEMENTATION="rmw_fastrtps_cpp",
        RCUTILS_LOGGING_USE_STDOUT="0",
    )
    name = "unresponsive_" + uuid.uuid4().hex
    code = """
import sys, rclpy
rclpy.init()
node = rclpy.create_node(sys.argv[1], namespace='/health_test')
print('READY', flush=True)
# Deliberately advertise parameter services without spinning their executor.
sys.stdin.buffer.read(1)
node.destroy_node()
rclpy.shutdown()
"""
    process = subprocess.Popen(
        [python, "-u", "-c", code, name],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    ready = queue.Queue()
    reader = threading.Thread(target=lambda: ready.put(process.stdout.readline()), daemon=True)
    reader.start()
    block = ROSHealth(
        parameters={
            "enabled": True,
            "domain_id": 219,
            "timeout_seconds": 1,
            "required_services": [
                {
                    "name": f"/health_test/{name}/get_parameters",
                    "type": "rcl_interfaces/srv/GetParameters",
                    "check_response": True,
                }
            ],
        }
    )
    try:
        assert ready.get(timeout=10) == b"READY\r\n"
        output = check(block)
        row = output["report"]["resources"][0]
        assert output["connected"] and output["alarm"]
        assert row["service_called"] is True and row["response_confirmed"] is False
        # A second timeout must clean up pending RPCs rather than accumulating them.
        assert check(block)["alarm"]
    finally:
        block.close()
        if process.poll() is None:
            try:
                process.stdin.write(b"x")
                process.stdin.flush()
                process.wait(timeout=5)
            except OSError, subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        reader.join(timeout=2)
        for pipe in (process.stdin, process.stdout, process.stderr):
            pipe.close()
