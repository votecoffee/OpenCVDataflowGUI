"""Saved visual groups and viewport metadata, independent of processing topology."""

import copy
import math
import uuid


def validate_layout(groups, viewport, blocks):
    if not isinstance(groups, list) or len(groups) > 10000:
        raise ValueError("Workspace groups must be a list with at most 10,000 entries")
    ids, members = set(), set()
    for group in groups:
        if (
            not isinstance(group, dict)
            or set(group) != {"id", "name", "members"}
            or not isinstance(group["id"], str)
            or not group["id"]
            or group["id"] in ids
            or not isinstance(group["name"], str)
            or not 1 <= len(group["name"]) <= 200
            or not isinstance(group["members"], list)
            or not group["members"]
            or any(
                not isinstance(m, str) or m not in blocks or m in members for m in group["members"]
            )
            or len(set(group["members"])) != len(group["members"])
        ):
            raise ValueError("Invalid workspace group or duplicate/missing member")
        ids.add(group["id"])
        members.update(group["members"])
    if viewport:
        if (
            not isinstance(viewport, dict)
            or set(viewport) != {"center", "zoom"}
            or not isinstance(viewport["center"], list)
            or len(viewport["center"]) != 2
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1e8
                for v in viewport["center"]
            )
            or type(viewport["zoom"]) not in (int, float)
            or not 0.005 <= viewport["zoom"] <= 3
        ):
            raise ValueError("Invalid workspace viewport")
    elif viewport != {}:
        raise ValueError("Workspace viewport must be an object")
    return copy.deepcopy(groups), copy.deepcopy(viewport)


def add_group(graph, members, name):
    group = {"id": uuid.uuid4().hex, "name": name.strip(), "members": list(members)}
    groups, _ = validate_layout([*graph.groups, group], graph.viewport, graph.blocks)
    graph.groups = groups
    return group
