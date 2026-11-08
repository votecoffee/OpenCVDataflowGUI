"""Check curved annotations, invalid map samples, and ticket/frame isolation."""

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.graphics_layers import GraphicLayer, GraphicsBundle, graphic_object
from opencv_dataflow.payloads import ValuePacket, frame
from opencv_dataflow.tickets import TicketClock

_RED = (0, 0, 255)


def setup_map():
    source = frame([30, 20], "camera")
    strip = frame([7, 3], "strip")
    clock = TicketClock()
    ticket = clock.issue("camera")
    strip_payload = {"spatial": strip, "ticket": ticket}
    y, x = np.indices((3, 7), dtype=np.float32)
    data = {
        "image": ValuePacket(np.zeros((20, 30), np.uint8), {"spatial": source, "ticket": ticket}),
        "map_x": ValuePacket(x + 3, strip_payload),
        "map_y": ValuePacket(y + 3 + x**2 / 3, strip_payload),
        "sampling": ValuePacket({
            "model": "path_strip_v1", "source_spatial": source, "strip_spatial": strip,
        }, strip_payload),
    }
    block = builtin_registry().create("inspect.map_strip_graphics")
    block.exposed_outputs.append("result_graphics")
    return block, data, strip_payload, clock


def test_map_densifies_a_curved_outline_and_interpolates_a_cropped_label():
    tool, data, payload, _ = setup_map()
    outline = graphic_object("polyline", [(0, 1), (6, 1)], _RED, thickness=3)
    crop = {**payload["spatial"], "image_size": [4, 3],
            "to_root": [[1, 0, 2], [0, 1, 0], [0, 0, 1]]}
    label = graphic_object("label", [(1.5, 1)], _RED, text="Narrow")
    bundle = GraphicsBundle((
        GraphicLayer("Width", (outline,), payload),
        GraphicLayer("Label", (label,), {**payload, "spatial": crop}),
    ))
    result = tool.execute({**data, "graphics": bundle})
    assert result is not None, tool.error
    assert result.outputs["counts"] == {"mapped": 2, "clipped": 0, "skipped": 0}
    line, text = result.outputs["result_graphics"].layers
    assert line.payload["spatial"] == data["image"].payload["spatial"]
    assert line.payload["ticket"] == payload["ticket"]
    assert line.objects[0].points[::6] == ((3, 4), (6, 7), (9, 16))
    assert line.objects[0].thickness == 3
    assert text.objects[0].points[0] == pytest.approx((6.5, 4 + (9 + 16) / 6))
    assert text.objects[0].text == "Narrow"


def test_map_splits_invalid_portion_and_does_not_join_a_closed_outline_across_it():
    tool, data, payload, _ = setup_map()
    data["map_x"].value[:, 3] = -1
    data["map_y"].value[:, 3] = -1
    outline = graphic_object("polyline", [(0, 1), (6, 1), (6, 2), (0, 2)], _RED, closed=True)
    bundle = GraphicsBundle((GraphicLayer("Closed outline", (outline,), payload),))
    result = tool.execute({**data, "graphics": bundle})
    assert result is not None, tool.error
    assert result.outputs["counts"] == {"mapped": 1, "clipped": 1, "skipped": 0}
    layer, = result.outputs["result_graphics"].layers
    assert len(layer.objects) == 3
    assert all(not obj.closed for obj in layer.objects)
    for obj in layer.objects:
        xs = [point[0] for point in obj.points]
        assert max(xs) <= 5 or min(xs) >= 7


def test_map_skips_stale_foreign_and_unsupported_graphics():
    tool, data, payload, clock = setup_map()
    old_ticket = clock.issue("camera")
    mark = graphic_object("marker", [(2, 1)], _RED)
    arrow = graphic_object("arrow", [(1, 1), (2, 1)], _RED)
    bundle = GraphicsBundle((
        GraphicLayer("Old", (mark,), {**payload, "ticket": old_ticket}),
        GraphicLayer("Foreign", (mark,), {"spatial": frame([7, 3], "different")}),
        GraphicLayer("Current", (mark, arrow), payload),
    ))
    result = tool.execute({**data, "graphics": bundle})
    assert result is not None, tool.error
    assert result.outputs["counts"] == {"mapped": 1, "clipped": 0, "skipped": 3}
    layer, = result.outputs["result_graphics"].layers
    assert layer.name == "Current" and len(layer.objects) == 1


def test_map_rejects_incompatible_map_grid_and_mixed_tickets():
    tool, data, payload, clock = setup_map()
    bundle = GraphicsBundle(())
    wrong_grid = ValuePacket(data["map_y"].value, {**payload, "spatial": frame([7, 3], "other")})
    assert tool.execute({**data, "map_y": wrong_grid, "graphics": bundle}) is None
    assert "sampling strip frame" in tool.error
    wrong_ticket = ValuePacket(data["map_y"].value, {**payload, "ticket": clock.issue("camera")})
    assert tool.execute({**data, "map_y": wrong_ticket, "graphics": bundle}) is None
    assert "share one ticket" in tool.error


def test_map_caps_samples_before_densifying_very_long_lines():
    tool, data, payload, _ = setup_map()
    outline = graphic_object("polyline", [(0, 1), (100_000, 1)], _RED)
    bundle = GraphicsBundle((GraphicLayer("Long", (outline,), payload),))
    assert tool.execute({**data, "graphics": bundle}) is None
    assert "sample budget" in tool.error
