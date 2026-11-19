"""Named immutable calibration snapshots and independently tagged capture grids."""

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .core import Block, Port, Result
from .parameter_validation import ValidationError
from .payloads import check_spatial
from .tickets import validate_ticket

REVISION = "calibration_revision"
CAPTURE = "calibration_capture"
APPLIED = "applied_calibration"
SCHEMA = "calibration_revision_v1"
KEYS = ("profile_id", "revision", "sha256")
SELECTOR_DEFAULTS = {"profile_id": "", "revision": ""}
SELECTOR_HELP = {
    "profile_id": "Named profile in the calibration directory. Blank opens a single JSON file.",
    "revision": "Exact revision to select. Required with a profile ID.",
}
MAX_REVISION_BYTES = 2 * 1024 * 1024
_RESERVED_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"{p}{i}" for p in ("com", "lpt") for i in range(10)),
}


def check_name(value, field):
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,47}", value)
        or value in _RESERVED_NAMES
    ):
        raise ValidationError(
            "Use 1 to 48 lowercase letters, digits, underscores or hyphens", field
        )


def check_selector(p):
    if p["profile_id"] != "" or p["revision"] != "":
        check_name(p["profile_id"], "profile_id")
        check_name(p["revision"], "revision")


def profile_path(block):
    p = block.parameters
    path = Path(p["path"])
    if p["path_base"] == "project" and not path.is_absolute():
        if block.project_dir is None:
            raise ValidationError(
                "Save or open the project before using relative calibration paths"
            )
        path = block.project_dir / path
    if p["profile_id"]:
        path = path / p["profile_id"] / f"{p['revision']}.json"
    return path


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def check_sha256(value, field="sha256"):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValidationError("Use an exact lowercase SHA-256 digest", field)


def check_frame(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise ValidationError("Specify a camera frame of 1 to 128 characters", "camera_frame")


def check_ref(value):
    if not isinstance(value, dict) or set(value) != set(KEYS):
        raise ValidationError("Malformed calibration revision reference")
    check_name(value["profile_id"], "profile_id")
    check_name(value["revision"], "revision")
    check_sha256(value["sha256"])
    return value


def bare_camera(profile):
    return {key: value for key, value in profile.items() if key != REVISION}


def bare_profile(profile):
    raw = bare_camera(profile)
    if raw.get("model") == "stereo_rig_v1":
        raw["left"] = bare_camera(raw["left"])
        raw["right"] = bare_camera(raw["right"])
    return raw


def check_binding(binding):
    if not isinstance(binding, dict) or set(binding) != {*KEYS, "camera_frame", "camera_sha256"}:
        raise ValidationError("Malformed camera calibration revision binding")
    check_ref({key: binding[key] for key in KEYS})
    check_frame(binding["camera_frame"])
    check_sha256(binding["camera_sha256"], "camera_sha256")
    return binding


def camera_binding(model):
    value = model.get(REVISION)
    if REVISION not in model:
        return None
    binding = check_binding(value)
    if binding["camera_frame"] != model.get("frame") or binding["camera_sha256"] != digest(
        bare_camera(model)
    ):
        raise ValidationError("Camera calibration changed after its revision was selected")
    return binding


def rig_revision(rig):
    if REVISION not in rig:
        return None
    ref = check_ref(rig[REVISION])
    matches = revision_payload(bare_profile(rig), ref["profile_id"], ref["revision"])
    bindings = [camera_binding(rig[side]) for side in ("left", "right")]
    if matches["sha256"] != ref["sha256"] or any(
        binding is None or any(binding[k] != ref[k] for k in KEYS) for binding in bindings
    ):
        raise ValidationError("Stereo calibration changed after its revision was selected")
    return ref


def revision_payload(profile, profile_id, revision):
    data = {"model": SCHEMA, "profile_id": profile_id, "revision": revision, "profile": profile}
    return {**data, "sha256": digest(data)}


def validate_revision_profile(profile):
    from .calibration import validate_profile

    if not isinstance(profile, dict) or profile.get("model") not in (
        "camera_model_v1",
        "stereo_rig_v1",
    ):
        raise ValidationError("Versioned profiles require a camera model or stereo rig")
    validate_profile(profile)
    if profile["model"] == "stereo_rig_v1":
        rig_revision(profile)
        cameras = [profile[side] for side in ("left", "right")]
        if cameras[0]["frame"] == cameras[1]["frame"]:
            raise ValidationError("Versioned rig cameras must have distinct optical frames")
    else:
        camera_binding(profile)
        cameras = [profile]
    for value in cameras:
        check_frame(value.get("frame"))


def make_revision(profile, profile_id, revision):
    check_name(profile_id, "profile_id")
    check_name(revision, "revision")
    validate_revision_profile(profile)
    return revision_payload(bare_profile(profile), profile_id, revision)


def read_revision(path, profile_id, revision):
    with path.open("rb") as input_file:
        raw = input_file.read(MAX_REVISION_BYTES + 1)
    if len(raw) > MAX_REVISION_BYTES:
        raise ValidationError("Calibration revision exceeds the 2 MiB size limit")

    def unique_fields(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValidationError("Calibration revision has duplicate JSON fields")
            value[key] = item
        return value

    data = json.loads(raw, object_pairs_hook=unique_fields)
    if (
        not isinstance(data, dict)
        or set(data) != {"model", *KEYS, "profile"}
        or data["model"] != SCHEMA
        or data["profile_id"] != profile_id
        or data["revision"] != revision
    ):
        raise ValidationError("Stored calibration does not match the selected profile and revision")
    profile = data["profile"]
    validate_revision_profile(profile)
    if data != revision_payload(bare_profile(profile), profile_id, revision):
        raise ValidationError("Calibration revision checksum mismatch")
    ref = {key: data[key] for key in KEYS}

    def bind(value):
        return {
            **value,
            REVISION: {**ref, "camera_frame": value["frame"], "camera_sha256": digest(value)},
        }

    if profile["model"] == "camera_model_v1":
        return bind(profile)
    return {**profile, REVISION: ref, **{side: bind(profile[side]) for side in ("left", "right")}}


def write_revision(path, revision):
    content = (json.dumps(revision, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if len(content) > MAX_REVISION_BYTES:
        raise ValidationError("Calibration revision exceeds the 2 MiB size limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = NamedTemporaryFile(prefix=".calibration-", dir=path.parent, delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            stream.write(content)
        if os.name == "nt":
            temporary.rename(path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def is_full_grid(payload, size):
    check_spatial(payload, size)
    spatial = payload.get("spatial")
    return (
        spatial is not None
        and spatial["root_size"] == list(size)
        and np.allclose(spatial["to_root"], np.eye(3), atol=1e-9, rtol=0)
        and payload.get("pyramid", {}).get("level", 0) == 0
    )


def require_raw_grid(payload, size):
    if not is_full_grid(payload, size) or APPLIED in payload:
        raise ValidationError("Capture must use the original full calibration pixel grid")


def check_capture(binding, payload, size):
    capture = payload.get(CAPTURE)
    if binding is None and CAPTURE not in payload:
        return
    if binding is None or capture is None:
        raise ValidationError("A versioned calibration requires matching capture metadata")
    check_binding(binding)
    if (
        not isinstance(capture, dict)
        or set(capture) != {*KEYS, "camera_frame", "ticket", "spatial"}
        or any(capture[k] != binding[k] for k in ("profile_id", "revision", "camera_frame"))
        or capture["sha256"] not in ("", binding["sha256"])
    ):
        raise ValidationError("Capture camera frame or calibration revision does not match")
    if capture["ticket"] != payload.get("ticket"):
        raise ValidationError("Calibration capture metadata belongs to a different ticket")
    require_raw_grid(payload, size)
    if capture["spatial"] != payload["spatial"]:
        raise ValidationError("Calibration capture metadata belongs to a different pixel grid")


def applied_payload(payload, binding):
    if binding is not None:
        payload.pop(CAPTURE, None)
        payload[APPLIED] = copy.deepcopy(binding)
    return payload


class DeclareCalibrationCapture(Block):
    type_id, title = "calibration.declare_capture", "Declare capture calibration revision"
    category, block_type, subtype = "Calibration", "Calibration", "Calibration profiles"
    inputs = (Port("image"),)
    outputs = (Port("image"),)
    defaults = {
        "profile_id": "camera-profile",
        "revision": "r1",
        "camera_frame": "camera",
        "sha256": "",
    }
    parameter_help = {
        "profile_id": "Profile used by this source at capture time, from an independent record.",
        "revision": "Revision used when the camera acquired this image.",
        "camera_frame": "Optical frame of the camera that acquired this image.",
        "sha256": "Optional SHA-256 from the recorded revision manifest.",
    }
    process_inputs_readonly = parallel_safe = True

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        check_name(p["profile_id"], "profile_id")
        check_name(p["revision"], "revision")
        check_frame(p["camera_frame"])
        if p["sha256"] != "":
            check_sha256(p["sha256"])

    def process(self, inputs):
        image = inputs["image"]
        payload = copy.deepcopy(self.input_payloads["image"])
        require_raw_grid(payload, image.shape[1::-1])
        ticket = payload.get("ticket")
        if ticket is not None:
            validate_ticket(ticket)
        declaration = {key: self.parameters[key] for key in (*KEYS, "camera_frame")}
        declaration["ticket"] = ticket
        declaration["spatial"] = copy.deepcopy(payload["spatial"])
        if CAPTURE in payload and payload[CAPTURE] != declaration:
            raise ValidationError("An existing capture calibration declaration cannot be changed")
        payload[CAPTURE] = declaration
        return Result({"image": image}, payloads={"image": payload})
