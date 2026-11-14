"""Opt-in native exact-time TF, interpolation, clock-reset and cancellation checks."""

import os
import time
import uuid
from contextlib import contextmanager

import numpy as np
import pytest

from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import NO_VALUE, ValuePacket
from opencv_dataflow.ros2_blocks import ROSBlock
from opencv_dataflow.ros2_client import ROSClient, ROSRemoteError
from opencv_dataflow.ros2_context import AttachAcquisition, RunIdentity
from opencv_dataflow.ros2_tf import LookupROSTF

pytestmark = pytest.mark.skipif(
    os.environ.get("OPENCV_DATAFLOW_TEST_ROS2") != "1",
    reason="Opt-in localhost Lyrical TF runtime required",
)


def transform(parent, child, sec, xyz, rotate=False):
    return {
        "header": {"stamp": {"sec": sec, "nanosec": 0}, "frame_id": parent},
        "child_frame_id": child,
        "transform": {
            "translation": dict(zip(("x", "y", "z"), xyz)),
            "rotation": {
                "x": 0.0,
                "y": 0.0,
                "z": float(np.sqrt(0.5)) if rotate else 0.0,
                "w": float(np.sqrt(0.5)) if rotate else 1.0,
            },
        },
    }


def publish(client, value, *, wait_for_ack=True):
    result = client.request(
        "publish",
        5,
        value=value,
        timeout=1,
        require_subscriber=True,
        wait_for_ack=wait_for_ack,
        stamp_mode="preserve",
    )
    assert result["published"]


@contextmanager
def runtime(sim=False):
    prefix = "/opencv_dataflow/tf_test_" + uuid.uuid4().hex
    params = {
        **LookupROSTF.defaults,
        "enabled": True,
        "tf_topic": prefix + "/tf",
        "tf_static_topic": prefix + "/static",
        "clock_topic": prefix + "/clock",
        "use_sim_time": sim,
        "max_age_seconds": 0.0,
        "timeout_seconds": 0.2,
    }
    clients = []
    try:
        listener = ROSClient(
            {**params, "kind": "tf", "role": "lookup", "publication_mode": "asynchronous"}, 30
        )
        clients.append(listener)
        for suffix, message, durability in (
            ("tf", "tf2_msgs/msg/TFMessage", "volatile"),
            ("static", "tf2_msgs/msg/TFMessage", "transient_local"),
            ("clock", "rosgraph_msgs/msg/Clock", "volatile"),
        ):
            client = ROSClient(
                {
                    **ROSBlock.defaults,
                    "kind": "data",
                    "role": "write",
                    "message_type": message,
                    "topic": prefix + "/" + suffix,
                    "durability": durability,
                },
                30,
            )
            clients.append(client)
        yield params, listener, clients[1], clients[2], clients[3]
    finally:
        for client in clients:
            client.close()
        assert all(c.process.poll() is not None for c in clients)
        assert all(not t.is_alive() for c in clients for t in c._threads)


def query(client, sec=15, *, target="base", source="camera", age=0.0):
    return client.request(
        "lookup",
        2,
        target=target,
        source=source,
        stamp={"sec": sec, "nanosec": 0},
        timeout=0.2,
        expected_epoch=client.info["tf_epoch"],
        max_age=age,
        future_tolerance=0.05,
    )


def populate(dynamic, static):
    publish(static, {"transforms": [transform("tool", "camera", 0, [0.0, 0.0, 0.5], True)]})
    publish(
        dynamic,
        {
            "transforms": [
                transform("base", "tool", 10, [0.0, 0.0, 0.0]),
                transform("base", "tool", 20, [2.0, 0.0, 0.0]),
            ]
        },
    )


def test_exact_interpolation_chain_direction_and_no_latest_fallback():
    with runtime() as (_, listener, dynamic, static, _clock):
        populate(dynamic, static)
        result = query(listener)
        translation = result["message"]["transform"]["translation"]
        np.testing.assert_allclose(list(translation.values()), [1, 0, 0.5])
        assert result["message"]["header"]["stamp"] == {"sec": 15, "nanosec": 0}
        assert result["info"]["query_mode"] == "exact" and not result["info"]["age_checked"]
        inverse = query(listener, target="camera", source="base")
        np.testing.assert_allclose(
            list(inverse["message"]["transform"]["translation"].values()), [0, 1, -0.5], atol=1e-12
        )
        with pytest.raises(ROSRemoteError) as exc:
            query(listener, sec=25)
        assert exc.value.code == "TFUnavailable"
        with pytest.raises(ROSRemoteError, match="zero/latest"):
            query(listener, sec=0)


def test_static_chain_samples_at_query_time_without_inventing_measurement_time():
    with runtime() as (_, listener, _dynamic, static, _clock):
        publish(static, {"transforms": [transform("base", "camera", 0, [1.0, 2.0, 3.0])]})
        result = query(listener)
        assert result["info"]["stamp_semantics"] == "query_time_not_source_measurement"
        assert result["info"]["source_sample_times_available"] is False
        assert result["info"]["tf2_header"]["stamp"]["sec"] in (0, 15)
        assert result["message"]["header"]["stamp"] == {"sec": 15, "nanosec": 0}


def test_clock_age_limits_and_backward_jump_are_explicit():
    with runtime(sim=True) as (_, listener, _dynamic, static, clock):
        publish(static, {"transforms": [transform("base", "camera", 0, [1.0, 2.0, 3.0])]})
        publish(clock, {"clock": {"sec": 100, "nanosec": 0}})
        end = time.monotonic() + 2
        while True:
            try:
                result = query(listener, sec=99, age=2)
                break
            except ROSRemoteError:
                if time.monotonic() > end:
                    raise
        assert result["info"]["age_seconds"] == 1
        with pytest.raises(ROSRemoteError, match="too old"):
            query(listener, sec=97, age=2)
        with pytest.raises(ROSRemoteError, match="ahead"):
            query(listener, sec=101, age=2)
        publish(clock, {"clock": {"sec": 50, "nanosec": 0}})
        end = time.monotonic() + 2
        while True:
            try:
                query(listener)
            except ROSRemoteError as exc:
                assert exc.code == "TFClockChanged"
                break
            if time.monotonic() > end:
                pytest.fail("Backward clock jump did not invalidate the TF epoch")


def test_host_context_guard_and_cache_replacement_after_reset():
    with runtime() as (params, _listener, dynamic, static, _clock):
        block = LookupROSTF(parameters={**params, "target_frame": "base", "unavailable": "wait"})
        run = RunIdentity(parameters={"run_id": "native-test"})
        current = run.execute({}).outputs["context"]
        image = ImageArray(
            np.ones((2, 3), np.uint8),
            "gray",
            {"ros2": {"header": {"stamp": {"sec": 15, "nanosec": 0}, "frame_id": "camera"}}},
        )
        attach = AttachAcquisition(parameters={"capture_mode": "header_stamp"})

        def tagged(context):
            result = attach.execute({"value": image, "context": context})
            return ValuePacket(result.outputs["value"], result.payloads["value"])

        try:
            old = tagged(current)
            assert block.execute({"value": old, "context": current}).outputs["pose"] is NO_VALUE
            first = block._client
            populate(dynamic, static)
            result = block.execute({"value": old, "context": current})
            assert result is not None, block.error
            np.testing.assert_allclose(result.outputs["pose"]["translation"], [1, 0, 0.5])
            assert result.outputs["info"]["context_checked"]
            new_context = run.execute({"reset": True}).outputs["context"]
            assert block.execute({"value": old, "context": new_context}) is None
            fresh = tagged(new_context)
            assert (
                block.execute({"value": fresh, "context": new_context}).outputs["pose"] is NO_VALUE
            )
            assert first.process.poll() == 0 and block._client is not first
            # Static TF must replay from transient-local history after replacement.
            # Re-send only dynamic samples; do not republish retained static data
            # while the departed reliable reader is leaving DDS discovery.
            # Dynamic TF is a stream, not retained history. Feed it during bounded
            # rediscovery; matching the fixture's other listener alone says nothing
            # about this replacement subscriber receiving a particular sample.
            end = time.monotonic() + 3
            while True:
                publish(
                    dynamic,
                    {
                        "transforms": [
                            transform("base", "tool", sec, [x, 0.0, 0.0])
                            for sec, x in ((10, 0.0), (20, 2.0))
                        ]
                    },
                    wait_for_ack=False,
                )
                replayed = block.execute({"value": fresh, "context": new_context})
                assert replayed is not None, block.error
                if replayed.outputs["pose"] is not NO_VALUE:
                    break
                assert time.monotonic() < end, "TF did not recover static/dynamic subscriptions"
            np.testing.assert_allclose(replayed.outputs["pose"]["translation"], [1, 0, 0.5])
        finally:
            process = block._client.process if block._client else None
            block.close()
            assert process is None or process.poll() == 0


def test_cancelled_lookup_reaps_the_native_worker():
    with runtime() as (_, listener, _dynamic, _static, _clock):
        started = time.monotonic()
        with pytest.raises(InterruptedError):
            listener.request(
                "lookup",
                12,
                lambda: time.monotonic() - started > 0.1,
                target="missing",
                source="also_missing",
                stamp={"sec": 17, "nanosec": 0},
                timeout=10,
                expected_epoch=0,
                max_age=0,
                future_tolerance=0,
            )
        assert listener.process.poll() is not None
