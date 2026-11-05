"""Save and apply a block's persisted configuration independently of its project."""

from copy import copy
from pathlib import Path
from urllib.parse import quote

from .core import Graph
from .project_xml import export_block_xml, read_block_xml


def block_xml_suffix(type_id):
    return f".{quote(type_id, safe='')}.xml"


def save_block_xml(block, path):
    suffix = block_xml_suffix(block.type_id)
    if not Path(path).name.lower().endswith(suffix.lower()):
        raise ValueError(f"Block XML filenames must end in {suffix}")
    export_block_xml(block, path)


def load_block_xml(graph, block_id, path, registry, *, owner=None, confirm_disconnect=None):
    target = graph.blocks[block_id]
    item = read_block_xml(path)
    file_type = item.get("type")
    if file_type != target.type_id:
        raise ValueError(f"Block type must match {target.type_id}. File type is {file_type!r}")
    if not isinstance(item.get("name"), str) or not item["name"].strip():
        raise ValueError("Block name must be nonempty text")
    if not isinstance(item.get("parameters"), dict):
        raise ValueError("Block parameters must be an object")
    definition = {
        "schema_version": 1,
        "blocks": [{**item, "id": target.id}],
        "connections": [],
    }
    staged = Graph.from_dict(definition, registry, project_dir=graph.project_dir)
    candidate = staged.blocks[block_id]
    candidate.position = list(target.position)
    try:
        candidate.validate_parameters()
        broken, remaining = [], []
        for wire in graph.connections:
            if block_id in (wire.source, wire.target):
                source = candidate if wire.source == block_id else graph.blocks[wire.source]
                dest = candidate if wire.target == block_id else graph.blocks[wire.target]
                try:
                    graph.validate_connection_ports(source, wire.output, dest, wire.input)
                except ValueError:
                    broken.append(wire)
                    continue
                if wire.target == block_id and wire.input == "model" and candidate.recipe_model:
                    broken.append(wire)
                    continue
            remaining.append(wire)
        if owner is not None:
            preview = copy(owner)
            preview.graph = copy(graph)
            preview.graph.blocks = {**graph.blocks, block_id: candidate}
            preview.graph.connections = remaining
            if preview.validate_interface() != (owner.inputs, owner.outputs):
                raise ValueError("Block XML would change the containing ToolBlock's public ports")
        if broken:
            if confirm_disconnect is None:
                raise ValueError("Loading this block requires approval to disconnect wires")
            if not confirm_disconnect(broken):
                staged.close()
                return None
        graph.recipes.merge(staged.recipes)
    except Exception:
        staged.close()
        raise

    graph.blocks[block_id] = candidate
    if broken:
        graph.connections = remaining
    graph.bind_recipes()
    graph.bind_activity()
    graph.bind_ticket_results()
    graph.invalidate()
    try:
        target.close()
    except Exception:
        candidate.logger.exception("Could not close the replaced block")
    return candidate
