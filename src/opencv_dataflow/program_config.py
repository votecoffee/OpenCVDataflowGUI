"""Persisted program lifecycle definitions; validation never starts a process."""

import copy
import json
import math
import re
from pathlib import PurePosixPath

HOOKS = ("startup", "before_run", "after_run", "shutdown")
DEFAULT_PROGRAM = {
    "name": "",
    "description": "",
    "enabled": False,
    "variables": {},
    "scripts": {name: "" for name in HOOKS},
    "script_timeout_seconds": 10.0,
    "services": [],
    "assets": [],
}
DEFAULT_SERVICE = {
    "id": "service",
    "enabled": True,
    "runtime": "application",
    "argv": [],
    "cwd": "{project_dir}",
    "environment": {},
    "ready_text": "",
    "startup_timeout_seconds": 30.0,
    "shutdown_timeout_seconds": 3.0,
    "stop_input": "",
}


def json_object(value, label, maximum=65536):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    if len(json.dumps(value, allow_nan=False).encode("utf-8")) > maximum:
        raise ValueError(f"{label} exceeds {maximum} bytes")
    return copy.deepcopy(value)


def _text(value, label, maximum):
    if not isinstance(value, str) or "\0" in value or len(value.encode("utf-8")) > maximum:
        raise ValueError(f"{label} must be text without NUL, at most {maximum} bytes")


def _seconds(value, label, maximum):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0.05 <= value <= maximum:
        raise ValueError(f"{label} must be between 0.05 and {maximum} seconds")


def validate_program(value):
    if not isinstance(value, dict) or value.keys() - DEFAULT_PROGRAM.keys():
        raise ValueError("Unknown program setting")
    result = copy.deepcopy(DEFAULT_PROGRAM)
    result.update(copy.deepcopy(value))
    for name, limit in (("name", 256), ("description", 16384)):
        _text(result[name], f"Program {name}", limit)
    if type(result["enabled"]) is not bool:
        raise ValueError("Program enabled must be boolean")
    result["variables"] = json_object(result["variables"], "Program variables")
    scripts = result["scripts"]
    if not isinstance(scripts, dict) or scripts.keys() - set(HOOKS):
        raise ValueError("Unknown program script event")
    result["scripts"] = {**DEFAULT_PROGRAM["scripts"], **scripts}
    for event, source in result["scripts"].items():
        _text(source, f"{event} script", 65536)
    _seconds(result["script_timeout_seconds"], "Script timeout", 120)
    services = result["services"]
    if not isinstance(services, list) or len(services) > 8:
        raise ValueError("A program supports up to eight managed services")
    ids, normalized = set(), []
    for service in services:
        if not isinstance(service, dict) or service.keys() - DEFAULT_SERVICE.keys():
            raise ValueError("Unknown managed service setting")
        service = {**copy.deepcopy(DEFAULT_SERVICE), **service}
        name = service["id"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", name):
            raise ValueError("Service ID must start with a letter and use letters/digits/_/-")
        if name in ids:
            raise ValueError(f"Duplicate service ID: {name}")
        ids.add(name)
        if type(service["enabled"]) is not bool or service["runtime"] not in (
            "application",
            "ros2",
        ):
            raise ValueError("Invalid service enabled/runtime setting")
        argv = service["argv"]
        if not isinstance(argv, list) or not 1 <= len(argv) <= 64:
            raise ValueError(f"Service {name} needs 1..64 command arguments")
        for argument in argv:
            _text(argument, "Command argument", 8192)
        if not argv[0].strip():
            raise ValueError("Service executable must not be empty")
        for field in ("cwd", "ready_text", "stop_input"):
            _text(service[field], f"Service {field}", 4096)
        environment = service["environment"]
        if not isinstance(environment, dict) or len(environment) > 64:
            raise ValueError("Service environment must have at most 64 entries")
        for key, item in environment.items():
            if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError("Invalid environment variable name")
            _text(item, "Environment value", 8192)
        _seconds(service["startup_timeout_seconds"], "Service startup timeout", 120)
        _seconds(service["shutdown_timeout_seconds"], "Service shutdown timeout", 15)
        normalized.append(service)
    result["services"] = normalized
    assets = result["assets"]
    if not isinstance(assets, list) or len(assets) > 128:
        raise ValueError("Program assets must be a list of up to 128 relative file paths")
    for asset in assets:
        _text(asset, "Program asset", 4096)
        path = PurePosixPath(asset.replace("\\", "/"))
        if not asset or path.is_absolute() or ".." in path.parts or ":" in asset:
            raise ValueError("Program assets must stay inside the project directory")
    return result


def validate_scripts(program):
    for event, source in program["scripts"].items():
        compile(source, f"<program:{event}>", "exec")
