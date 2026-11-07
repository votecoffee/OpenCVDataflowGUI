"""Catalog-wide guard against missing shared metadata and persistence support."""

import json

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.graphics_layers import RESULT_GRAPHICS
from opencv_dataflow.project_xml import export_project_xml, read_project_xml


def test_builtin_catalog_defaults_help_and_persistence(tmp_path):
    registry = builtin_registry()
    catalog = []
    for kind, cls in registry.types.items():
        block = registry.create(kind, block_id=kind)
        block.validate_parameters()
        input_names = {port.name for port in block.inputs}
        output_names = {port.name for port in block.output_ports}
        assert cls.parameter_choices.keys() <= cls.defaults.keys(), kind
        assert cls.parameter_ranges.keys() <= cls.defaults.keys(), kind
        assert cls.parameter_help.keys() <= cls.defaults.keys(), kind
        assert cls.input_help.keys() <= input_names, kind
        if block.layout_only:
            assert not input_names and not output_names, kind
        else:
            assert RESULT_GRAPHICS in output_names, kind
        assert set(block.exposed_inputs) <= input_names, kind
        assert set(block.exposed_outputs) <= output_names, kind
        if block.ticket_source:
            assert "ticket" in input_names and "ticket" in output_names, kind
        item = block.to_project_item()
        catalog.append(item)
        definition = {"schema_version": 1, "blocks": [item], "connections": []}
        restored = Graph.from_dict(json.loads(json.dumps(definition)), registry)
        assert restored.to_dict() == definition, kind
    expected = {"schema_version": 1, "blocks": catalog, "connections": []}
    xml_path = tmp_path / "catalog.xml"
    export_project_xml(expected, xml_path)
    assert read_project_xml(xml_path) == expected
