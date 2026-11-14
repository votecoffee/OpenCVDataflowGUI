"""Host TF state/cancellation contracts without optional ROS or network access."""

import copy

import numpy as np

from opencv_dataflow import ros2_tf
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket
from opencv_dataflow.ros2_client import ROSRemoteError
from opencv_dataflow.ros2_context import AttachAcquisition, RunIdentity
from opencv_dataflow.ros2_poses import pose_from_ros
from opencv_dataflow.ros2_tf import LookupROSTF


def image():
    return ImageArray(
        np.ones((2, 3), np.uint8),
        "gray",
        {"ros2": {"header": {"stamp": {"sec": 17, "nanosec": 1}, "frame_id": "camera"}}},
    )


class Connection:
    def __init__(self, *args):
        self.info, self.error, self.closed, self.calls = {"tf_epoch": 0}, None, False, []

    def request(self, operation, *args, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if self.error:
            raise self.error
        return {
            "message": {
                "header": {"stamp": kwargs["stamp"], "frame_id": kwargs["target"]},
                "child_frame_id": kwargs["source"],
                "transform": {
                    "translation": {"x": 1.0, "y": 0.0, "z": 0.0},
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
            },
            "info": {"tf_epoch": 0, "query_mode": "exact"},
        }

    def close(self):
        self.closed = True


def test_disabled_tf_does_not_start_ros_and_zero_queries_are_rejected(monkeypatch):
    opened = []
    monkeypatch.setattr(ros2_tf, "ROSClient", lambda *args: opened.append(True))
    block = LookupROSTF()
    assert block.execute({"value": image()}).outputs["pose"] is NO_VALUE
    block.parameters["enabled"] = True
    value = image()
    value.metadata["ros2"]["header"]["stamp"] = {"sec": 0, "nanosec": 0}
    assert block.execute({"value": value}) is None
    assert not opened


def test_tf_preserves_value_payload_and_explicit_source_target_direction(monkeypatch):
    client = Connection()
    monkeypatch.setattr(ros2_tf, "ROSClient", lambda *args: client)
    block = LookupROSTF(parameters={"enabled": True, "units": "mm", "source_frame": "tool0"})
    result = block.execute({"value": image()})
    assert result is not None, block.error
    assert client.calls[0]["source"] == "tool0" and client.calls[0]["target"] == "base_link"
    assert result.outputs["pose"]["source_frame"] == "tool0"
    assert result.outputs["pose"]["translation"] == [1000, 0, 0]
    np.testing.assert_array_equal(result.outputs["value"], image())
    block.close()
    assert client.closed


def test_unavailable_wait_and_clock_reset_latch_are_distinct(monkeypatch):
    client = Connection()
    monkeypatch.setattr(ros2_tf, "ROSClient", lambda *args: client)
    block = LookupROSTF(parameters={"enabled": True, "unavailable": "wait"})
    client.error = ROSRemoteError("missing transform", "TFUnavailable")
    assert block.execute({"value": image()}).outputs["pose"] is NO_VALUE
    assert not client.closed
    client.error = ROSRemoteError("clock changed", "TFClockChanged")
    assert block.execute({"value": image()}) is None
    assert client.closed and block._reset_required
    calls = len(client.calls)
    client.error = None
    assert block.execute({"value": image()}) is None and len(client.calls) == calls
    block.reset_runtime()
    assert block.execute({"value": image()}).outputs["pose"]
    block.close()


def test_stale_context_does_not_start_tf_and_current_scope_change_restarts_it(monkeypatch):
    opened = []

    def connection(*args):
        opened.append(Connection())
        return opened[-1]

    monkeypatch.setattr(ros2_tf, "ROSClient", connection)
    run = RunIdentity()
    first = run.execute({}).outputs["context"]
    attach = AttachAcquisition(parameters={"capture_mode": "header_stamp"})

    def tagged(context):
        result = attach.execute({"value": image(), "context": context})
        return ValuePacket(result.outputs["value"], result.payloads["value"])

    old = tagged(first)
    current = run.execute({"reset": True}).outputs["context"]
    block = LookupROSTF(parameters={"enabled": True})
    assert block.execute({"value": old, "context": current}) is None and not opened
    assert block.execute({"value": old, "context": first}) is not None
    assert block.execute({"value": tagged(current), "context": current}) is not None
    assert len(opened) == 2 and opened[0].closed
    block.close()


def test_cancelled_lookup_closes_worker_and_emits_no_event(monkeypatch):
    client = Connection()
    client.error = InterruptedError()
    monkeypatch.setattr(ros2_tf, "ROSClient", lambda *args: client)
    block = LookupROSTF(parameters={"enabled": True})
    assert block.execute({"value": image()}).outputs["pose"] is NO_VALUE
    assert client.closed


def test_chained_pose_lookup_uses_its_parent_frame_over_original_image_metadata(monkeypatch):
    client = Connection()
    monkeypatch.setattr(ros2_tf, "ROSClient", lambda *args: client)
    pose, _ = pose_from_ros(
        {
            "header": {"stamp": {"sec": 17, "nanosec": 1}, "frame_id": "base_link"},
            "child_frame_id": "tool0",
            "transform": {
                "translation": {"x": 0.0, "y": 0.0, "z": 1.0},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        }
    )
    block = LookupROSTF(parameters={"enabled": True, "target_frame": "world"})
    result = block.execute({"value": ValuePacket(pose, image().metadata)})
    assert result is not None, block.error
    assert client.calls[0]["source"] == "base_link"
    block.close()
