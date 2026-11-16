"""Strict JSON loading and explicit publication of motion trajectory assets."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

from .core import Block, Port, Result
from .motion_trajectory import MAX_BYTES, validate_trajectory
from .parameter_validation import ValidationError


def trajectory_bytes(value):
    validate_trajectory(value)
    raw = (json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2) + "\n").encode()
    if len(raw) > MAX_BYTES:
        raise ValidationError("Formatted trajectory exceeds 1 MiB")
    return raw


def unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate trajectory JSON field: " + key)
        result[key] = value
    return result


def trajectory_path(block):
    name = block.parameters["path"]
    if (
        not isinstance(name, str)
        or not name.strip()
        or "\0" in name
        or Path(name).suffix.lower() != ".json"
    ):
        raise ValidationError("Choose a trajectory .json path", "path")
    path = Path(name)
    if not path.is_absolute():
        if block.project_dir is None:
            raise ValidationError("Save/open the project or use an absolute path", "path")
        path = block.project_dir / path
    return path.resolve()


class LoadMotionTrajectory(Block):
    type_id, title = "sequence.load_trajectory", "Load motion trajectory"
    category = block_type = "Sequence"
    subtype = "Recorded motion"
    outputs = (Port("trajectory", "data"), Port("info", "data"))
    defaults = {"path": "", "expected_sha256": ""}
    project_asset_parameters = (*Block.project_asset_parameters, "path")
    parameter_help = {
        "path": "Strict UTF-8 JSON, up to 1 MiB; relative to the saved project.",
        "expected_sha256": "Optional exact-byte SHA256; Save As copies this sealed asset.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        digest = p["expected_sha256"]
        if not isinstance(digest, str) or (
            digest and (len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest))
        ):
            raise ValidationError(
                "Expected SHA256 must be empty or 64 hex digits", "expected_sha256"
            )

    def process(self, inputs):
        path = trajectory_path(self)
        with path.open("rb") as stream:
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValidationError("Trajectory exceeds 1 MiB", "path")
        digest = hashlib.sha256(raw).hexdigest()
        if (
            self.parameters["expected_sha256"]
            and digest != self.parameters["expected_sha256"].lower()
        ):
            raise ValidationError("Trajectory file differs from expected bytes", "expected_sha256")
        try:
            value = json.loads(
                raw.decode("utf8"),
                object_pairs_hook=unique_fields,
                parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)),
            )
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise ValidationError("Invalid strict trajectory JSON: " + str(exc), "path") from exc
        records = validate_trajectory(value)
        return Result(
            {
                "trajectory": value,
                "info": {
                    "sha256": digest,
                    "bytes": len(raw),
                    "trajectory_id": value["trajectory_id"],
                    "clip_id": value["clip_id"],
                    "frames": len(records),
                },
            },
            message="Loaded motion trajectory: " + path.name,
        )


class SaveMotionTrajectory(Block):
    type_id, title = "output.motion_trajectory", "Save motion trajectory"
    category = block_type = "Output"
    subtype = "Recorded motion"
    preview_safe = False
    inputs = (Port("trajectory", "data"),)
    outputs = (Port("info", "data"),)
    defaults = {"path": "motion_trajectory.json", "write_mode": "new"}
    parameter_choices = {"write_mode": ("new", "replace")}
    parameter_help = {
        "path": "Destination .json, relative to the saved project; directory must exist.",
        "write_mode": "New preserves existing files; Replace publishes atomically.",
    }

    def process(self, inputs):
        raw = trajectory_bytes(inputs["trajectory"])
        path = trajectory_path(self)
        if self.parameters["write_mode"] == "new":
            with path.open("xb") as stream:
                stream.write(raw)
        else:
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=path.parent, prefix=".trajectory-", suffix=".tmp", delete=False
                ) as stream:
                    temporary = Path(stream.name)
                    stream.write(raw)
                os.replace(temporary, path)
                temporary = None
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return Result(
            {
                "info": {
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes": len(raw),
                    "trajectory_id": inputs["trajectory"]["trajectory_id"],
                }
            },
            message="Saved motion trajectory: " + path.name,
        )
