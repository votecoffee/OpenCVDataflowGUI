"""Portable failure codes/messages carried on ticketed connections."""

import json


def normalize_reasons(value):
    if isinstance(value, (str, dict)):
        value = [value]
    if not isinstance(value, list) or len(value) > 32:
        raise ValueError("Fail reasons must be a list of at most 32 codes/messages")
    result = []
    for item in value:
        if isinstance(item, str):
            item = {"code": item, "message": item}
        if (
            not isinstance(item, dict)
            or not set(item) <= {"code", "message"}
            or not isinstance(item.get("code"), str)
            or not 1 <= len(item["code"]) <= 128
            or not isinstance(item.get("message", item["code"]), str)
            or len(item.get("message", item["code"])) > 512
        ):
            raise ValueError(
                "Fail reason requires a code (1..128) and message (up to 512 characters)"
            )
        item = {"code": item["code"], "message": item.get("message", item["code"])}
        if item not in result:
            result.append(item)
    if len(json.dumps(result).encode()) > 4096:
        raise ValueError("Fail reasons exceed 4 KiB")
    return result


def merge_reasons(*collections):
    result = []
    for collection in collections:
        for reason in normalize_reasons(collection):
            if reason not in result:
                trial = result + [reason]
                if len(trial) <= 32 and len(json.dumps(trial).encode()) <= 4096:
                    result = trial
    return result


def reasons_in(value):
    from .outcomes import packets_in
    from .payloads import metadata

    result = normalize_reasons(metadata(value).get("outcome", {}).get("fail_reasons", []))
    for packet in packets_in(value):
        result = merge_reasons(result, packet.payload.get("outcome", {}).get("fail_reasons", []))
    return result
