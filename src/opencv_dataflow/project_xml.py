"""Versioned XML snapshots and field-level comparisons of saved project data."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from xml.etree import ElementTree as ET

ROOT = "opencv-dataflow-project"
BLOCK_ROOT = "opencv-dataflow-block"
XML_FILTER = "Project XML (*.xml)"
SCALARS = {str: "string", bool: "boolean", int: "integer", float: "number", type(None): "null"}


def _encode(node, value):
    if isinstance(value, dict):
        node.set("type", "object")
        for key, child in value.items():
            # JSON escaping also preserves keys with XML-forbidden characters.
            _encode(ET.SubElement(node, "field", name=json.dumps(key)[1:-1]), child)
    elif isinstance(value, list):
        node.set("type", "array")
        for child in value:
            _encode(ET.SubElement(node, "item"), child)
    else:
        node.set("type", SCALARS[type(value)])
        node.text = json.dumps(value, allow_nan=False)


def export_project_xml(project, path):
    """Export the same persisted values as Graph.save, without changing the project."""
    _write_xml(ROOT, project, path)


def export_block_xml(block, path):
    _write_xml(BLOCK_ROOT, block.to_project_item(), path)


def _write_xml(root_name, data, path):
    root = ET.Element(root_name, version="1")
    _encode(root, json.loads(json.dumps(data, allow_nan=False)))
    ET.indent(root)
    path = Path(path)
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=".project-xml-", delete=False) as stream:
            temporary = Path(stream.name)
            ET.ElementTree(root).write(stream, encoding="utf-8", xml_declaration=True)
            stream.write(b"\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class _NoDTD(ET.TreeBuilder):
    def doctype(self, name, public_id, system_id):
        raise ValueError("Project XML cannot contain a DOCTYPE declaration")


def _decode(node):
    kind = node.get("type")
    if kind in ("object", "array"):
        if (node.text or "").strip() or any((child.tail or "").strip() for child in node):
            raise ValueError("Project XML containers cannot contain scalar text")
        if kind == "array":
            if any(child.tag != "item" for child in node):
                raise ValueError("Project XML arrays must contain items")
            return [_decode(child) for child in node]
        fields = {}
        for child in node:
            if child.tag != "field" or "name" not in child.attrib:
                raise ValueError("Project XML objects must contain named fields")
            name = json.loads('"' + child.attrib["name"] + '"')
            if name in fields:
                raise ValueError(f"Duplicate project XML field {name!r}")
            fields[name] = _decode(child)
        return fields
    if len(node):
        raise ValueError("Project XML scalar values cannot contain elements")
    value = json.loads(node.text or "")
    if SCALARS.get(type(value)) != kind or kind is None:
        raise ValueError(f"Invalid project XML value of type {kind!r}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Project XML numbers must be finite")
    return value


def read_project_xml(path):
    return _read_xml(ROOT, path)


def read_block_xml(path):
    return _read_xml(BLOCK_ROOT, path)


def _read_xml(root_name, path):
    parser = ET.XMLParser(target=_NoDTD())
    root = ET.parse(path, parser=parser).getroot()
    if root.tag != root_name or root.get("version") != "1" or root.get("type") != "object":
        raise ValueError(f"Expected a version 1 {root_name} XML file")
    return _decode(root)


@dataclass(frozen=True)
class ProjectChange:
    path: str
    kind: str
    before: object = None
    after: object = None


def _path(parent, key):
    token = str(key).replace("~", "~0").replace("/", "~1")
    return f"{parent}/{token}"


def _block_index(values):
    if not isinstance(values, list):
        return None
    indexed = {}
    for block in values:
        if not isinstance(block, dict) or not isinstance(block.get("id"), str):
            return None
        if block["id"] in indexed:
            return None
        indexed[block["id"]] = block
    return indexed


def _changes(before, after, path=""):
    if isinstance(before, dict) and isinstance(after, dict):
        graph = {"schema_version", "blocks", "connections"} <= before.keys() & after.keys()
        for key in sorted(before.keys() | after.keys()):
            child = _path(path, key)
            if key not in before:
                yield ProjectChange(child, "Added", after=after[key])
            elif key not in after:
                yield ProjectChange(child, "Removed", before=before[key])
            elif graph and key == "blocks":
                old_blocks, new_blocks = _block_index(before[key]), _block_index(after[key])
                if old_blocks is None or new_blocks is None:
                    yield from _changes(before[key], after[key], child)
                    continue
                old_order = [key for key in old_blocks if key in new_blocks]
                new_order = [key for key in new_blocks if key in old_blocks]
                if old_order != new_order:
                    yield ProjectChange(child, "Reordered", old_order, new_order)
                yield from _changes(old_blocks, new_blocks, child)
            else:
                yield from _changes(before[key], after[key], child)
    elif isinstance(before, list) and isinstance(after, list):
        for index in range(max(len(before), len(after))):
            child = _path(path, index)
            if index >= len(before):
                yield ProjectChange(child, "Added", after=after[index])
            elif index >= len(after):
                yield ProjectChange(child, "Removed", before=before[index])
            else:
                yield from _changes(before[index], after[index], child)
    elif type(before) is not type(after) or before != after or (
        isinstance(before, float) and math.copysign(1, before) != math.copysign(1, after)
    ):
        yield ProjectChange(path, "Changed", before, after)


def compare_project_xml(before, after):
    return list(_changes(read_project_xml(before), read_project_xml(after)))
