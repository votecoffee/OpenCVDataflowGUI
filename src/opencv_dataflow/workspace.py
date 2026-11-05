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


def arranged_positions(graph, sizes):
    """Lay out dependency columns with separate lanes for groups and independent flows."""
    parents = {key: set() for key in graph.blocks}
    children = {key: set() for key in graph.blocks}
    for wire in graph.connections:
        parents[wire.target].add(wire.source)
        children[wire.source].add(wire.target)
    ranks = {}
    for key in graph.ordered():
        ranks[key] = 1 + max((ranks[parent] for parent in parents[key]), default=-1)

    # Groups also tie disconnected members into the same visual component.
    neighbors = {key: parents[key] | children[key] for key in graph.blocks}
    groups = {}
    for group in graph.groups:
        first, *others = group["members"]
        neighbors[first].update(others)
        for key in others:
            neighbors[key].add(first)
        groups.update(dict.fromkeys(group["members"], group["id"]))

    insertion_order = {key: index for index, key in enumerate(graph.blocks)}
    unseen = set(graph.blocks)
    positions, top = {}, 0
    for first in graph.blocks:
        if first not in unseen:
            continue
        component, pending = [], [first]
        unseen.remove(first)
        while pending:
            key = pending.pop()
            component.append(key)
            adjacent = neighbors[key] & unseen
            unseen.difference_update(adjacent)
            pending.extend(adjacent)

        lanes, widths = {}, {}
        for key in sorted(component, key=insertion_order.__getitem__):
            rank = ranks[key]
            lanes.setdefault(groups.get(key), {}).setdefault(rank, []).append(key)
            widths[rank] = max(widths.get(rank, 0), sizes[key][0])
        lefts, left = {}, 0
        for rank in sorted(widths):
            lefts[rank] = left
            left += widths[rank] + 110

        for columns in lanes.values():
            _order_columns(columns, parents, children)
            heights = {
                rank: sum(sizes[key][1] for key in column) + 48 * (len(column) - 1)
                for rank, column in columns.items()
            }
            lane_height = max(heights.values())
            for rank, column in columns.items():
                y = top + (lane_height - heights[rank]) / 2
                for key in column:
                    positions[key] = (lefts[rank], y)
                    y += sizes[key][1] + 48
            top += lane_height + 90
        top += 30
    return positions


def _order_columns(columns, parents, children):
    """Use neighboring row positions to keep branches in a consistent order."""
    ranks = sorted(columns)
    rows = {key: index for column in columns.values() for index, key in enumerate(column)}
    for order, neighbors in ((ranks, parents), (reversed(ranks), children)):
        for rank in order:
            column = columns[rank]

            def order_key(key):
                nearby = [rows[other] for other in neighbors[key] if other in rows]
                return (sum(nearby) / len(nearby) if nearby else rows[key], rows[key])

            column.sort(key=order_key)
            rows.update({key: index for index, key in enumerate(column)})
