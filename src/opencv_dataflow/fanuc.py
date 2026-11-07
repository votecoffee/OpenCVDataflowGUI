"""FANUC Workbench reads, reviewed register changes and calibration poses."""

import json
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .core import Block, Port, Result
from .geometry3d_common import pose_matrix, pose_record
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE
from .visual_style import INDUSTRIAL

_TARGET = re.compile(r"R\[[1-9][0-9]*\]|PR\[[1-8],[1-9][0-9]*\]")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PLAN_ID = re.compile(r"[0-9a-f]{32}")
_AXES = "XYZWPR"
_GROUP = NumericRange(1, 8, whole=True)
_FRAME_INDEX = NumericRange(1, 255, whole=True)
_FRAME_NUMBER = NumericRange(0, 255, whole=True)


def workbench_command():
    if os.name != "nt":
        raise RuntimeError("FANUC online blocks require the Windows FANUC Workbench")
    home = os.environ.get("OPENCV_DATAFLOW_FANUC_HOME")
    if home:
        directories = (Path(home).resolve(),)
    else:
        local = os.environ.get("LOCALAPPDATA")
        if not local:
            raise RuntimeError("Set OPENCV_DATAFLOW_FANUC_HOME to the installed Workbench folder")
        root = Path(local) / "Programs" / "FanucMiddleware"
        directories = (root / "1.0.1", root / "1.0.0")
    for directory in directories:
        executable = directory / ".venv" / "Scripts" / "python.exe"
        script = directory / "scripts" / "fanuc_middleware.py"
        if executable.is_file() and script.is_file():
            return [str(executable), "-I", "-B", str(script)]
    raise RuntimeError("Set OPENCV_DATAFLOW_FANUC_HOME to the installed Workbench folder")


def workbench_call(profile, operation, arguments, timeout):
    command = workbench_command()
    with TemporaryDirectory(prefix="dataflow-fanuc-") as folder:
        request = Path(folder) / "request.json"
        request.write_text(json.dumps(arguments, allow_nan=False), encoding="utf-8")
        try:
            response = subprocess.run(
                [*command, "run", operation, "--config", str(profile), "--request", str(request)],
                capture_output=True, encoding="utf-8", errors="replace",
                stdin=subprocess.DEVNULL, timeout=timeout, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"FANUC {operation} timed out; check Workbench receipts") from exc
    try:
        envelope = json.loads(response.stdout)
    except ValueError as exc:
        raise RuntimeError(f"FANUC {operation} returned no JSON receipt") from exc
    if response.returncode or not envelope.get("ok"):
        error = envelope.get("error") or {}
        message = str(error.get("message", f"exit code {response.returncode}"))[:512]
        raise RuntimeError(f"FANUC {operation}: {message}")
    if not isinstance(envelope.get("result"), dict):
        raise RuntimeError(f"FANUC {operation} returned an invalid receipt")
    return envelope["result"]


class FanucBlock(Block):
    category = block_type = INDUSTRIAL
    subtype = "FANUC"
    preview_safe = False
    requires_serial_execution = True
    validation_skip_output = True
    defaults = {"profile": "", "robot_id": "", "timeout_seconds": 15.0}
    parameter_ranges = {"timeout_seconds": NumericRange(1, 120)}
    parameter_help = {"profile": "Local Workbench JSON policy for the configured robot."}

    def target(self):
        target = self.parameters["target"]
        if not isinstance(target, str) or len(target) > 32 or not _TARGET.fullmatch(target):
            raise ValidationError("Use R[index] or PR[group,index]", "target")
        return target

    def robot(self):
        value = self.parameters["robot_id"]
        if not isinstance(value, str) or not value.strip():
            raise ValidationError("Configure the Workbench robot ID", "robot_id")
        return value.strip()

    def call(self, operation, **arguments):
        value = self.parameters["profile"]
        if not isinstance(value, str) or not value.strip():
            raise ValidationError("Select a local FANUC Workbench profile", "profile")
        path = (self.project_dir or Path.cwd()) / Path(value)
        path = path.resolve()
        if not path.is_file():
            raise ValidationError("The FANUC Workbench profile does not exist", "profile")
        return workbench_call(path, operation, arguments, float(self.parameters["timeout_seconds"]))


class ReadFanucRegister(FanucBlock):
    type_id, title = "industrial.fanuc_register", "Read FANUC register"
    inputs = (Port("trigger", "any", required=False, required_when_connected=True),)
    outputs = (Port("value", "any"), Port("reference", "data"))
    defaults = {**FanucBlock.defaults, "target": "", "enabled": False}
    parameter_help = {**FanucBlock.parameter_help, "target": "Use R[index] or PR[group,index]."}

    def process(self, inputs):
        if self.parameters["enabled"] is not True:
            return Result({"value": NO_VALUE, "reference": NO_VALUE}, message="FANUC read disabled")
        robot, target = self.robot(), self.target()
        reference = self.call("core.registers.read", robot_id=robot, target=target)
        if reference.get("robot_id") != robot or reference.get("target") != target:
            raise RuntimeError("Workbench register receipt does not match the requested target")
        register = reference["value"]
        value = register["value"] if target.startswith("R[") else register
        return Result({"value": value, "reference": reference}, message=f"Read {target}")


class ReadFanucPosition(FanucBlock):
    type_id, title = "industrial.fanuc_position", "Read FANUC current position"
    inputs = ReadFanucRegister.inputs
    outputs = (
        Port("position", "data"), Port("user_frame", "integer"), Port("user_tool", "integer"),
    )
    defaults = {**FanucBlock.defaults, "group": 1, "enabled": False}
    parameter_ranges = {**FanucBlock.parameter_ranges, "group": _GROUP}

    def process(self, inputs):
        if self.parameters["enabled"] is not True:
            return Result(
                {port.name: NO_VALUE for port in self.outputs}, message="FANUC read disabled"
            )
        group = _GROUP.check(self.parameters["group"], "Motion group", "group")
        status = self.call(
            "core.controller.status", robot_id=self.robot(), group=group, alarm_limit=0,
        )
        position = status.get("position")
        if not isinstance(position, dict) or position.get("initialized") is not True:
            raise RuntimeError("Workbench did not return an initialized current position")
        return Result({
            "position": {**position, "group": group, "timestamp": status["timestamp"]},
            "user_frame": position["user_frame"],
            "user_tool": position["user_tool"],
        })


class PrepareFanucRegister(FanucBlock):
    type_id, title = "industrial.fanuc_prepare_register", "Prepare FANUC register write"
    inputs = (Port("reference", "data"), Port("change", "data"), Port("enable", "boolean"))
    outputs = (Port("review", "data"),)
    defaults = {**FanucBlock.defaults, "target": "", "enabled": False}
    parameter_help = ReadFanucRegister.parameter_help
    input_help = {
        "reference": "Preserved reference output of a matching FANUC register read.",
        "change": 'Numeric register {"type":"integer","value":42} or PR {"values":{"X":12}}.',
    }

    def process(self, inputs):
        if self.parameters["enabled"] is not True or not inputs["enable"]:
            return Result({"review": NO_VALUE}, message="FANUC prepare disabled")
        robot, target = self.robot(), self.target()
        reference, change = inputs["reference"], inputs["change"]
        if (
            not isinstance(reference, dict) or reference.get("kind") != "register"
            or reference.get("robot_id") != robot or reference.get("target") != target
            or not _SHA256.fullmatch(str(reference.get("sha256", "")))
            or not isinstance(reference.get("local_path"), str)
        ):
            raise ValidationError("Connect a retained read for the exact FANUC register target")
        if not isinstance(change, dict):
            raise ValidationError("Supply a FANUC register change object")
        review = self.call(
            "core.registers.update", robot_id=robot, target=target, change=change,
            expected_sha256=reference["sha256"], reference_path=reference["local_path"],
        )
        if review.get("status") != "prepared_only" or review.get("controller_written") is not False:
            raise RuntimeError("Workbench did not return a prepare-only review")
        return Result({"review": review}, message="FANUC register write prepared for review")


class CommitFanucRegister(FanucBlock):
    type_id, title = "industrial.fanuc_commit_register", "Commit approved FANUC register write"
    inputs = (Port("approval", "data"), Port("enable", "boolean"))
    outputs = (Port("receipt", "data"),)
    defaults = {**FanucBlock.defaults, "target": "", "write_enabled": False}
    parameter_help = ReadFanucRegister.parameter_help
    input_help = {
        "approval": "Operator-approved plan_id, review_sha256 and approved_sha256 for this target."
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._uncertain = False

    def reset_runtime(self):
        self._uncertain = False

    def copy_runtime_from(self, other):
        self._uncertain = other._uncertain

    def process(self, inputs):
        if self.parameters["write_enabled"] is not True or not inputs["enable"]:
            return Result({"receipt": NO_VALUE}, message="FANUC write disabled")
        if self._uncertain:
            raise ValidationError("Inspect the retained FANUC plan before resetting flow state")
        robot, target = self.robot(), self.target()
        approval = inputs["approval"]
        if (
            not isinstance(approval, dict)
            or set(approval) != {"plan_id", "review_sha256", "approved_sha256"}
            or not _PLAN_ID.fullmatch(str(approval["plan_id"]))
            or any(not _SHA256.fullmatch(str(approval[name])) for name in (
                "review_sha256", "approved_sha256"
            ))
        ):
            raise ValidationError("Supply an exact approved Workbench plan ID and hashes")
        status = self.call("change.status", plan_id=approval["plan_id"])
        review = status.get("review") or {}
        proposal = review.get("proposal") or {}
        if (
            status.get("status") != "prepared" or status.get("current_policy") is not True
            or status.get("review_sha256") != approval["review_sha256"]
            or review.get("proposed_sha256") != approval["approved_sha256"]
            or proposal.get("kind") != "register" or proposal.get("robot_id") != robot
            or proposal.get("target") != target
        ):
            raise ValidationError("Approval does not match the current FANUC register review")
        self._uncertain = True
        receipt = self.call("change.commit", **approval)
        if (
            receipt.get("status") != "verified"
            or receipt.get("plan_id") != approval["plan_id"]
            or receipt.get("robot_id") != robot
            or receipt.get("target") != target
        ):
            raise RuntimeError("FANUC write outcome was not verified; inspect the retained plan")
        self._uncertain = False
        return Result({"receipt": receipt}, message=f"Verified FANUC write to {target}")


def _cartesian_pose(values, source, target):
    if not isinstance(values, dict) or any(
        not isinstance(values.get(axis), (int, float, np.integer, np.floating))
        or isinstance(values[axis], (bool, np.bool_))
        for axis in _AXES
    ):
        raise ValidationError("FANUC Cartesian values require numeric X, Y, Z, W, P, R")
    x, y, z, w, pitch, r = (NumericRange().check(values[axis], axis) for axis in _AXES)
    angles = np.radians([w, pitch, r])
    cw, cp, cr = np.cos(angles)
    sw, sp, sr = np.sin(angles)
    rx = np.array([[1, 0, 0], [0, cw, -sw], [0, sw, cw]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return pose_record(pose_matrix(rz @ ry @ rx, [x, y, z]), source, target, "mm")


class _ReadFanucOffset(FanucBlock):
    inputs = (*ReadFanucRegister.inputs, Port("index", "integer", required=False))
    outputs = (Port("offset", "data"), Port("pose", "data", visual_type="pose"))
    defaults = {
        **FanucBlock.defaults, "group": 1, "index": 1, "enabled": False,
        "source_frame": "", "target_frame": "",
    }
    parameter_ranges = {**FanucBlock.parameter_ranges, "group": _GROUP, "index": _FRAME_INDEX}
    parameter_help = {
        **FanucBlock.parameter_help,
        "group": "Motion group containing the frame table.",
        "index": "Frame table entry. A wired index takes precedence.",
        "source_frame": "Optional pose source label. Blank uses the frame kind and index.",
        "target_frame": "Optional target label. Blank uses base for UFRAME or gripper for UTOOL.",
    }

    def process(self, inputs):
        if self.parameters["enabled"] is not True:
            return Result({"offset": NO_VALUE, "pose": NO_VALUE}, message="FANUC read disabled")
        p = self.parameters
        group = _GROUP.check(p["group"], "Motion group", "group")
        index = _FRAME_INDEX.check(inputs.get("index", p["index"]), "Frame index", "index")
        source, target = p["source_frame"], p["target_frame"]
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValidationError("Frame names must be text", "source_frame", "target_frame")
        source = source.strip() or f"{self.source_prefix}_{index}"
        target = target.strip() or self.default_target
        expression = f"{self.frame_kind}[{group},{index}]"
        robot = self.robot()
        receipt = self.call("core.controller.sample", robot_id=robot, expressions=[expression])
        if receipt.get("robot_id") != robot or receipt.get("provider") != "frrobot":
            raise RuntimeError("Workbench frame receipt does not match the configured robot")
        errors, values = receipt.get("errors"), receipt.get("values")
        if not isinstance(errors, dict) or not isinstance(values, dict):
            raise RuntimeError("Workbench returned an invalid frame receipt")
        if expression in errors:
            raise RuntimeError(f"Workbench could not read {expression} as an initialized offset")
        offset = values.get(expression)
        if (
            not isinstance(offset, dict)
            or type(offset.get("group")) is not int or type(offset.get("index")) is not int
            or (offset.get("kind"), offset.get("group"), offset.get("index"))
            != (self.frame_kind, group, index)
            or offset.get("initialized") is not True
            or offset.get("representation") != "cartesian"
        ):
            raise RuntimeError(f"Workbench did not return the requested Cartesian {expression}")
        pose = _cartesian_pose(offset.get("values"), source, target)
        timestamp = NumericRange(0).check(receipt.get("timestamp"), "Sample timestamp")
        offset = {**offset, "robot_id": robot, "timestamp": timestamp}
        return Result({"offset": offset, "pose": pose}, message=f"Read {expression}")


class ReadFanucUserFrame(_ReadFanucOffset):
    type_id, title = "industrial.fanuc_uframe", "Read FANUC UFRAME offset"
    frame_kind, source_prefix, default_target = "UFRAME", "user_frame", "base"


class ReadFanucUserTool(_ReadFanucOffset):
    type_id, title = "industrial.fanuc_utool", "Read FANUC UTOOL offset"
    frame_kind, source_prefix, default_target = "UTOOL", "user_tool", "gripper"


class FanucCartesianPose(Block):
    type_id, title = "industrial.fanuc_pose", "FANUC Cartesian pose"
    category = block_type = INDUSTRIAL
    subtype = "FANUC"
    inputs = (Port("position", "data"),)
    outputs = (Port("pose", "data", visual_type="pose"),)
    defaults = {
        "source_frame": "gripper", "target_frame": "base",
        "expected_user_frame": 0, "expected_user_tool": 0,
    }
    parameter_ranges = {
        "expected_user_frame": _FRAME_NUMBER,
        "expected_user_tool": _FRAME_NUMBER,
    }
    parameter_help = {
        "source_frame": "Name of the sampled tool frame. Use gripper for flange samples.",
        "target_frame": "Name of the sampled user frame. Use base for UF 0.",
        "expected_user_frame": "Reject a sample measured in any other user frame.",
        "expected_user_tool": "Reject a sample measured with any other user tool.",
    }

    def process(self, inputs):
        position = inputs["position"]
        p = self.parameters
        if (
            not isinstance(position, dict) or position.get("initialized") is not True
            or position.get("representation") != "cartesian"
        ):
            raise ValidationError("Use an initialized Cartesian FANUC XYZ/WPR sample")
        frame, tool = position.get("user_frame"), position.get("user_tool")
        if type(frame) is not int or type(tool) is not int:
            raise ValidationError("The FANUC sample needs integer user frame and tool numbers")
        expected = (p["expected_user_frame"], p["expected_user_tool"])
        if (frame, tool) != expected:
            raise ValidationError(
                f"The FANUC sample's user frame or tool does not match. "
                f"Read UF {frame}, UT {tool}. Expected UF {expected[0]}, UT {expected[1]}."
            )
        pose = _cartesian_pose(position.get("values"), p["source_frame"], p["target_frame"])
        return Result({"pose": pose})


FANUC_BLOCKS = (
    ReadFanucRegister, ReadFanucPosition, PrepareFanucRegister, CommitFanucRegister,
    FanucCartesianPose, ReadFanucUserFrame, ReadFanucUserTool,
)
