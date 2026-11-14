"""Offline frame directions, quaternion references and stale-generation guards."""

import copy
import json

import cv2
import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.geometry3d_common import pose_record
from opencv_dataflow.images import ImageArray
from opencv_dataflow.payloads import FlowGroup, ValuePacket
from opencv_dataflow.ros2_context import AttachAcquisition, GuardRunIdentity, RunIdentity, scope_key
from opencv_dataflow.ros2_poses import DecodeROSPose, EncodeROSPose, pose_from_ros, pose_to_ros
from opencv_dataflow.tickets import TicketClock, ticket_key
from opencv_dataflow.toolbox import block_entry

HEADER = {"stamp": {"sec": 17, "nanosec": 123456789}, "frame_id": "base_link"}


def message():
    return {
        "header": copy.deepcopy(HEADER),
        "child_frame_id": "tool0",
        "transform": {
            "translation": {"x": 0.1, "y": -0.2, "z": 0.3},
            "rotation": {"x": 0.0, "y": 0.0, "z": float(np.sqrt(0.5)), "w": float(np.sqrt(0.5))},
        },
    }


def test_parent_from_child_inverse_and_physical_units():
    pose, info = pose_from_ros(message())
    assert pose["source_frame"] == "tool0" and pose["target_frame"] == "base_link"
    np.testing.assert_allclose(
        np.array(pose["source_to_target"]) @ [1, 0, 0, 1], [0.1, 0.8, 0.3, 1], atol=1e-14
    )
    inverse, _ = pose_from_ros(message(), invert=True, units="mm")
    matrix = np.array(inverse["source_to_target"])
    matrix[:3, 3] *= 0.001
    np.testing.assert_allclose(matrix @ np.array(pose["source_to_target"]), np.eye(4), atol=1e-14)
    assert inverse["source_frame"] == "base_link" and inverse["target_frame"] == "tool0"
    assert inverse["ros2"]["header"]["frame_id"] == "tool0"
    assert inverse["ros2"]["source_header"]["frame_id"] == "base_link"
    assert info["quaternion_order"] == "xyzw"


@pytest.mark.parametrize(
    "axis,angle",
    [(0, 0), (0, np.pi), (1, np.pi), (2, np.pi), (0, np.pi - 1e-10), (1, 1.7), (2, -2.9)],
)
def test_quaternion_roundtrip_matches_independent_rodrigues(axis, angle):
    rotation = np.zeros(3)
    rotation[axis] = angle
    matrix = np.eye(4)
    matrix[:3, :3] = cv2.Rodrigues(rotation)[0]
    matrix[:3, 3] = [100, 200, -300]
    source = pose_record(matrix, "object", "camera", "mm", ros2={"header": HEADER})
    encoded = pose_to_ros(source)
    decoded, _ = pose_from_ros(encoded, units="mm")
    np.testing.assert_allclose(decoded["source_to_target"], matrix, atol=1e-8)
    assert encoded["header"]["frame_id"] == "camera"
    assert encoded["transform"]["translation"] == {"x": 0.1, "y": 0.2, "z": -0.3}


def test_pose_stamped_child_is_explicit_and_original_message_is_private():
    pose = pose_from_ros(message())[0]
    value = pose_to_ros(pose, message_kind="pose_stamped")
    decoded, _ = pose_from_ros(value, message_kind="pose_stamped", child_frame="tool0")
    np.testing.assert_allclose(decoded["source_to_target"], pose["source_to_target"], atol=1e-14)
    block = DecodeROSPose(parameters={"message_kind": "pose_stamped", "child_frame": "tool0"})
    result = block.execute({"message": value})
    assert result is not None, block.error
    result.outputs["message"]["header"]["frame_id"] = "other"
    assert value["header"]["frame_id"] == "base_link"
    assert EncodeROSPose().execute({"pose": pose}).outputs["message"]["child_frame_id"] == "tool0"


@pytest.mark.parametrize(
    "kind", ["zero", "nonunit", "boolean", "nonfinite", "frame", "expectation", "same_frame"]
)
def test_bad_pose_data_is_contained(kind):
    value, parameters = message(), {}
    if kind == "zero":
        value["transform"]["rotation"] = dict.fromkeys(("x", "y", "z", "w"), 0.0)
    elif kind == "nonunit":
        value["transform"]["rotation"]["w"] = 2.0
    elif kind == "boolean":
        value["transform"]["translation"]["x"] = True
    elif kind == "nonfinite":
        value["transform"]["rotation"]["x"] = np.inf
    elif kind == "frame":
        value["header"]["frame_id"] = "/base_link"
    elif kind == "expectation":
        parameters["expected_parent"] = "world"
    else:
        value["child_frame_id"] = value["header"]["frame_id"]
    block = DecodeROSPose(parameters=parameters)
    assert block.execute({"message": value}) is None
    assert not block.current_emissions


def test_normalization_requires_opt_in_and_unitless_pose_encoding_is_rejected():
    value = message()
    value["transform"]["rotation"]["w"] = 2.0
    pose, info = pose_from_ros(value, quaternion_policy="normalize")
    assert info["quaternion_norm"] > 2
    pose["units"] = "unitless"
    with pytest.raises(ValueError):
        pose_to_ros(pose)


def run():
    return RunIdentity(parameters={"dataset_id": "parts", "run_id": "batch-A"})


def tagged(context, frame="camera"):
    image = ImageArray(
        np.arange(12, dtype=np.uint8).reshape(3, 4),
        "gray",
        {"ros2": {"header": {"stamp": {"sec": 17, "nanosec": 123456789}, "frame_id": frame}}},
    )
    tag = AttachAcquisition(parameters={"capture_mode": "header_stamp"})
    result = tag.execute({"value": image, "context": context})
    assert result is not None, tag.error
    return ValuePacket(result.outputs["value"], result.payloads["value"])


def test_reset_edges_and_reopen_rotate_unsaved_runtime_identity():
    block = run()
    first = block.execute({}).outputs["context"]
    second = block.execute({"reset": True}).outputs["context"]
    assert second["generation"] == first["generation"] + 1
    assert block.execute({"reset": True}).outputs["context"] == second
    block.execute({"reset": False})
    assert block.execute({"reset": True}).outputs["context"]["generation"] == 2
    assert "session_id" not in json.dumps(block.to_project_item())
    block.reset_runtime()
    reopened = block.execute({}).outputs["context"]
    assert reopened["session_id"] != first["session_id"]
    assert scope_key(reopened) != scope_key(first)


def test_stamp_capture_correlates_frames_without_relabeling_them():
    current = run().execute({}).outputs["context"]
    camera, robot = tagged(current, "camera"), tagged(current, "base_link")
    assert ticket_key(camera.payload["ticket"]) == ticket_key(robot.payload["ticket"])
    assert camera.value.metadata["ros2"]["header"]["frame_id"] == "camera"
    assert robot.value.metadata["ros2"]["header"]["frame_id"] == "base_link"


def test_new_execution_graph_rotates_context_while_display_copy_only_mirrors_it():
    previous = run()
    old = previous.execute({}).outputs["context"]
    execution = run()
    execution.copy_runtime_from(previous)
    fresh = execution.execute({}).outputs["context"]
    assert fresh["session_id"] != old["session_id"]
    assert execution.execute({}).outputs["context"] == fresh
    previous.copy_runtime_from(execution)
    assert previous._context == fresh


def test_external_envelope_echoes_run_token_without_inventing_acquisition_metadata():
    block = run()
    old = block.execute({}).outputs["context"]
    current_result = block.execute({"reset": True})
    current = current_result.outputs["context"]
    assert current_result.outputs["run_token"] == scope_key(current)
    guard = GuardRunIdentity(parameters={"identity_source": "envelope"})
    assert guard.execute({"value": {"run_id": scope_key(old)}, "context": current}) is None
    result = guard.execute(
        {"value": {"run_id": scope_key(current), "request_id": "A"}, "context": current}
    )
    assert result is not None, guard.error
    assert result.outputs["info"]["capture_checked"] is False
    assert "acquisition" not in result.payloads.get("value", {})


def test_recipe_or_api_run_label_change_rotates_context_without_reusing_queued_identity():
    block = run()
    first = block.execute({}).outputs["context"]
    block.parameters["dataset_id"] = "another-dataset"
    current = block.execute({}).outputs["context"]
    assert (
        current["dataset_id"] == "another-dataset" and current["session_id"] != first["session_id"]
    )


def test_old_queued_result_and_mixed_generation_groups_are_rejected():
    block = run()
    before = block.execute({}).outputs["context"]
    queued = tagged(before)
    current = block.execute({"reset": True}).outputs["context"]
    fresh = tagged(current)
    guard = GuardRunIdentity()
    assert guard.execute({"value": queued, "context": current}) is None
    assert not guard.current_emissions
    assert (
        guard.execute({"value": fresh, "context": current}).outputs["info"]["checked_members"] == 1
    )
    assert (
        guard.execute({"value": FlowGroup((queued, fresh), ("old", "new")), "context": current})
        is None
    )
    assert (
        guard.execute({"value": FlowGroup((fresh, fresh), ("a", "b")), "context": current})
        is not None
    )


def test_missing_capture_is_not_current_and_integer_field_capture_is_exact():
    current = run().execute({}).outputs["context"]
    block = AttachAcquisition(parameters={"capture_mode": "field", "capture_field": "part.id"})
    value = {"part": {"id": 2**60 + 1}}
    result = block.execute({"value": value, "context": current})
    assert result.outputs["ticket"]["value"] == 2**60 + 1
    packet = ValuePacket(result.outputs["value"], result.payloads["value"])
    packet.payload["acquisition"].pop("capture")
    assert GuardRunIdentity().execute({"value": packet, "context": current}) is None
    assert GuardRunIdentity().execute({"value": value, "context": current}) is None


def test_parallel_child_ticket_can_retain_its_parent_acquisition_scope():
    current = run().execute({}).outputs["context"]
    value = tagged(current)
    value.payload["ticket"] = TicketClock("parallel-child").issue("inspection")
    assert GuardRunIdentity().execute({"value": value, "context": current}) is not None


def test_attach_cannot_wash_an_old_acquisition_into_the_current_run():
    block = run()
    first = block.execute({}).outputs["context"]
    old = tagged(first)
    current = block.execute({"reset": True}).outputs["context"]
    attach = AttachAcquisition(parameters={"capture_mode": "header_stamp"})
    assert attach.execute({"value": old, "context": current}) is None
    assert not attach.current_emissions


def test_ticket_capture_keeps_original_ticket_while_scoping_its_session():
    current = run().execute({}).outputs["context"]
    original = TicketClock().issue("plc", "external", "A-17")
    block = AttachAcquisition()
    value = ValuePacket({"score": 0.9}, {"ticket": original})
    result = block.execute({"value": value, "context": current})
    assert result.outputs["ticket"]["value"] == "A-17"
    assert result.outputs["ticket"]["session"] == scope_key(current)
    assert result.outputs["acquisition"]["original_ticket"] == original
    assert value.payload["ticket"] == original


def test_registry_xml_and_unsaved_context_contract(tmp_path):
    registry, graph = builtin_registry(), Graph()
    for kind in (
        "industrial.ros2_pose_decode",
        "industrial.ros2_pose_encode",
        "industrial.ros2_run",
        "industrial.ros2_acquisition",
        "industrial.ros2_run_guard",
        "industrial.ros2_tf",
    ):
        block = graph.add(registry.create(kind))
        block.validate_parameters()
        assert block_entry(block).path == ("Industrial integration", "ROS 2")
    path = tmp_path / "poses.xml"
    graph.save(path)
    restored = Graph.load(path, registry)
    assert len(restored.blocks) == 6 and "session_id" not in path.read_text()
    restored.close()
    graph.close()
