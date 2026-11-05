import json
from copy import deepcopy

import pytest

from opencv_dataflow.blocks import builtin_registry, demo_graph
from opencv_dataflow.project_xml import (
    ProjectChange,
    compare_project_xml,
    export_project_xml,
    read_project_xml,
)
from opencv_dataflow.toolblocks import encapsulate


def test_project_xml_contains_every_saved_field_and_preserves_types(tmp_path):
    registry = builtin_registry()
    graph = demo_graph(registry)
    block_ids = list(graph.blocks)
    nested = encapsulate(graph, block_ids[-2:], "Inspection", registry)
    nested.description = "Line 1\r\nLine 2"
    nested.documentation = "<p>Width &amp; orientation 📷</p>"
    inner = nested.graph.blocks[block_ids[-1]]
    inner.description = "Null \x00 and unpaired Unicode \ud800"
    graph.viewport = {"center": [12.5, 4.0], "zoom": 1.2}
    original = tmp_path / "program.json"
    graph.save(original)
    expected = json.loads(original.read_text(encoding="utf-8"))
    expected["extension"] = {'odd/"~\x00\t': [False, 0, -0.0, None, "", {}, []]}

    snapshot = tmp_path / "program.xml"
    export_project_xml(expected, snapshot)
    assert snapshot.read_bytes().startswith(b"<?xml version='1.0' encoding='utf-8'?>")
    restored = read_project_xml(snapshot)
    assert json.dumps(restored, sort_keys=True) == json.dumps(expected, sort_keys=True)
    assert not compare_project_xml(snapshot, snapshot)


def test_compare_pairs_blocks_by_id_and_reports_changes_in_nested_graphs(tmp_path):
    inner = {
        "schema_version": 1,
        "blocks": [{"id": "inner", "parameters": {"limit": 2, "mode": False}}],
        "connections": [],
    }
    before = {
        "schema_version": 1,
        "blocks": [{"id": "a"}, {"id": "job", "toolblock": {"graph": inner}}],
        "connections": [],
    }
    after = deepcopy(before)
    after["blocks"].reverse()
    after["blocks"].insert(1, {"id": "new"})
    parameters = after["blocks"][0]["toolblock"]["graph"]["blocks"][0]["parameters"]
    parameters["limit"] = 3
    parameters["mode"] = 0
    left, right = tmp_path / "before.xml", tmp_path / "after.xml"
    export_project_xml(before, left)
    export_project_xml(after, right)

    parameters_path = "/blocks/job/toolblock/graph/blocks/inner/parameters"
    assert compare_project_xml(left, right) == [
        ProjectChange("/blocks", "Reordered", ["a", "job"], ["job", "a"]),
        ProjectChange(f"{parameters_path}/limit", "Changed", 2, 3),
        ProjectChange(f"{parameters_path}/mode", "Changed", False, 0),
        ProjectChange("/blocks/new", "Added", after={"id": "new"}),
    ]
    assert compare_project_xml(right, left)[-1] == ProjectChange(
        "/blocks/new", "Removed", before={"id": "new"}
    )


def test_project_xml_refuses_doctype_and_duplicate_fields(tmp_path):
    path = tmp_path / "invalid.xml"
    path.write_text(
        '<!DOCTYPE opencv-dataflow-project [<!ENTITY test "unexpected">]>'
        '<opencv-dataflow-project version="1" type="object" />',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="DOCTYPE"):
        read_project_xml(path)
    path.write_text(
        '<opencv-dataflow-project version="1" type="object">'
        '<field name="x" type="integer">1</field><field name="x" type="integer">2</field>'
        '</opencv-dataflow-project>',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate"):
        read_project_xml(path)
