"""Per-block diagnostics, severity filters and docked offscreen inspection."""

import json
import logging

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

from opencv_dataflow.app import MainWindow
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Block, Graph, Port, Result
from opencv_dataflow.diagnostic_viewer import DiagnosticViewer
from opencv_dataflow.diagnostics import COLORS, LEVELS, level_name, parameters_snapshot
from opencv_dataflow.examples import parallel_parts_graph
from opencv_dataflow.payloads import ValuePacket
from opencv_dataflow.recipes import reference
from opencv_dataflow.tickets import TicketClock
from opencv_dataflow.toolblocks import Binding, ToolBlock


class DiagnosticBlock(Block):
    type_id, title = "custom.diagnostic_fixture", "Diagnostic fixture"
    inputs, outputs = (Port("image"),), (Port("image"),)
    defaults = {"threshold": 12, "options": {"values": [1, 2]}}

    def process(self, inputs):
        self.logger.info("Handled image %s", inputs["image"].shape)
        return Result({"image": inputs["image"]})


class Broken(DiagnosticBlock):
    def inner(self):
        raise KeyError("missing calibration")

    def process(self, inputs):
        try:
            self.inner()
        except KeyError as exc:
            raise RuntimeError("inspection failed") from exc


def records(block, kind="diagnostic"):
    return [e for e in block.logger.snapshot()["events"] if e["kind"] == kind]


def test_standalone_logger_and_early_messages_follow_block_into_graph():
    block = DiagnosticBlock()
    block.logger.info("Before binding")
    assert records(block)[0]["message"] == "Before binding"
    graph = Graph()
    graph.add(block)
    block.logger.info("After binding")
    graph.bind_activity()
    assert [e["message"] for e in records(block)] == ["Before binding", "After binding"]
    assert len(graph.activity.snapshot()["events"]) == 3


@pytest.mark.parametrize("level", list(LEVELS))
def test_all_levels_and_python_formatting(level):
    block = DiagnosticBlock()
    getattr(block.logger, level.lower())("Count %s", 3)
    event = records(block)[0]
    assert event["severity"] == level and event["message"] == "Count 3"
    assert block.logger.isEnabledFor(LEVELS[level])
    assert level_name(LEVELS[level]) == level
    assert event["exception"] is None


def test_independent_loggers_do_not_mix_events_or_mutate_parameter_snapshots():
    graph = Graph()
    a, b = graph.add(DiagnosticBlock()), graph.add(DiagnosticBlock())
    a.logger.warning("A")
    b.logger.error("B")
    a.parameters["options"]["values"].append(99)
    assert [e["message"] for e in records(a)] == ["A"]
    assert [e["message"] for e in records(b)] == ["B"]
    assert records(a)[0]["parameters"]["options"]["values"] == [1, 2]
    records(a)[0]["parameters"]["threshold"] = -1
    assert records(a)[0]["parameters"]["threshold"] == 12


def test_caught_exception_records_full_cause_chain_parameters_and_ticket():
    block = Broken()
    ticket = TicketClock().issue("camera")
    block.execute({"image": ValuePacket(np.zeros((12, 16), np.uint8), {"ticket": ticket})})
    event = records(block, "exception")[0]
    assert event["severity"] == "ERROR"
    assert event["parameters"]["threshold"] == 12 and event["tickets"] == [ticket]
    trace = event["exception"]["traceback"]
    assert "inner" in trace and "KeyError" in trace and "RuntimeError" in trace
    assert "direct cause" in trace and "missing calibration" in trace
    finish = records(block, "finish")[0]
    assert finish["exception"]["traceback"] == trace
    assert isinstance(event["exception"]["type"], str)
    json.dumps(block.logger.snapshot(), allow_nan=False)


def test_logger_exception_within_handled_except_does_not_change_tool_judgment():
    class Recovering(DiagnosticBlock):
        def process(self, inputs):
            try:
                raise ValueError("optional measurement")
            except ValueError:
                self.logger.exception("Recovered")
            return Result({"image": inputs["image"]})

    block = Recovering()
    block.execute({"image": np.zeros((12, 16), np.uint8)})
    assert block.state == "pass"
    assert "ValueError" in records(block, "exception")[0]["exception"]["traceback"]
    assert records(block, "finish")[0]["exception"] is None


def test_recipe_effective_parameters_are_preserved_in_diagnostic_and_finish():
    graph = Graph()
    block = graph.add(DiagnosticBlock())
    recipe = graph.recipes.put("Widget", {"threshold": 99})
    block.recipe_bindings = {"threshold": "settings.threshold"}
    graph.bind_recipes()
    block.execute(
        {
            "image": ValuePacket(
                np.zeros((12, 16), np.uint8),
                {
                    "recipe_ref": reference(recipe),
                },
            )
        }
    )
    assert records(block)[0]["parameters"]["threshold"] == 99
    assert records(block, "finish")[0]["parameters"]["threshold"] == 99
    assert block.parameters["threshold"] == 12


def test_large_parameters_images_and_tracebacks_are_bounded_with_explicit_markers():
    assert parameters_snapshot({"score": np.float32(0.5), "count": np.uint16(12)}) == {
        "score": 0.5,
        "count": 12,
    }
    result = parameters_snapshot({"array": np.zeros((100, 200), np.uint8)})
    assert result["array"] == {"shape": [100, 200], "dtype": "uint8", "data": "[not retained]"}
    block = DiagnosticBlock()
    block.logger.info("long", parameters={str(i): "x" * 4000 for i in range(100)})
    assert records(block)[0]["parameters"]["truncated"]
    try:
        raise RuntimeError("x" * 50000)
    except RuntimeError:
        block.logger.exception("Large error")
    event = records(block, "exception")[0]
    assert event["exception"]["truncated"] and "[truncated]" in event["exception"]["traceback"]


def test_default_exception_info_and_level_aliases():
    block = DiagnosticBlock()
    block.logger.log(logging.WARNING, "alias")
    block.logger.log("WARN", "alias")
    assert all(e["severity"] == "WARNING" for e in records(block))
    block.logger.exception("No current exception")
    assert records(block)[-1]["exception"] is None
    with pytest.raises(ValueError, match="Diagnostic level"):
        block.logger.log("unknown", "bad")


def viewer_events(view):
    return [
        view.table.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole)
        for i in range(view.table.topLevelItemCount())
    ]


def test_dock_is_at_bottom_and_follows_real_canvas_selection(app):
    window = MainWindow()
    graph = Graph()
    a, b = graph.add(DiagnosticBlock()), graph.add(DiagnosticBlock())
    window.replace_graph(graph)
    a.logger.info("A only")
    b.logger.warning("B only")
    window.diagnostics_action.trigger()
    assert not window.diagnostics_dock.isHidden()
    assert window.dockWidgetArea(window.diagnostics_dock) == Qt.DockWidgetArea.BottomDockWidgetArea
    window.diagnostics_view.refresh()
    assert len(viewer_events(window.diagnostics_view)) == 2
    window.scene.nodes[a.id].setSelected(True)
    assert [e["message"] for e in viewer_events(window.diagnostics_view)] == ["A only"]
    window.scene.clearSelection()
    assert len(viewer_events(window.diagnostics_view)) == 2
    window.diagnostics_dock.close()
    assert not window.diagnostics_action.isChecked()
    window.dirty = False
    window.close()


def test_severity_filter_colors_and_search_preserve_captured_verbosity(app):
    graph = Graph()
    block = graph.add(DiagnosticBlock())
    for name in LEVELS:
        block.logger.log(name, f"Message {name}")
    view = DiagnosticViewer(lambda: graph.activity)
    assert len(viewer_events(view)) == 4  # INFO and above
    view.severity.setCurrentText("ERROR")
    assert [e["severity"] for e in viewer_events(view)] == ["ERROR", "CRITICAL"]
    for index, level in enumerate(("ERROR", "CRITICAL")):
        assert view.table.topLevelItem(index).foreground(1).color() == QColor(COLORS[level])
    view.severity.setCurrentText("TRACE")
    assert {e["severity"] for e in viewer_events(view)} == set(LEVELS)
    view.search.setText("Message DEBUG")
    assert len(viewer_events(view)) == 1
    assert len(records(block)) == 6
    view.close()


def test_exception_detail_tabs_include_traceback_and_full_parameters(app):
    graph = Graph()
    block = graph.add(Broken())
    block.execute({"image": np.zeros((12, 16), np.uint8)})
    view = DiagnosticViewer(lambda: graph.activity)
    view.severity.setCurrentText("ERROR")
    event = next(e for e in viewer_events(view) if e["kind"] == "exception")
    item = next(
        view.table.topLevelItem(i)
        for i in range(view.table.topLevelItemCount())
        if view.table.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole)["id"] == event["id"]
    )
    view.table.setCurrentItem(item)
    assert "inspection failed" in view.details.toPlainText()
    assert json.loads(view.parameters.toPlainText())["threshold"] == 12
    assert "missing calibration" in view.traceback.toPlainText()
    assert "inner" in view.traceback.toPlainText() and view.tabs.currentIndex() == 2
    view.tabs.setCurrentIndex(1)
    block.logger.error("Another failure")
    view.refresh()
    assert view.tabs.currentIndex() == 1  # preserve the chosen detail tab on refresh
    view.search.setText("missing calibration")
    assert view.table.topLevelItemCount() == 2  # exception and error finish
    view.search.setText("not a matching exception")
    assert not view.traceback.toPlainText()
    view.close()


def test_paused_view_filters_frozen_data_and_resume_sees_new_events(app):
    graph = Graph()
    a, b = graph.add(DiagnosticBlock()), graph.add(DiagnosticBlock())
    a.logger.info("First")
    view = DiagnosticViewer(lambda: graph.activity)
    view.pause.setChecked(True)
    b.logger.warning("Second")
    view.refresh()
    assert len(viewer_events(view)) == 1
    view.set_block(b)
    assert not viewer_events(view)
    view.pause.setChecked(False)
    assert [e["message"] for e in viewer_events(view)] == ["Second"]
    view.close()


def test_identically_named_nested_blocks_are_isolated_by_logical_path(app):
    registry, outer = builtin_registry(), Graph()
    children = []
    for index in range(2):
        inner = Graph()
        child = inner.add(registry.create("cv.gray", block_id="same_id"))
        children.append(child)
        outer.add(
            ToolBlock(
                inner,
                {"image": Binding(child.id, "image")},
                {"image": Binding(child.id, "image")},
                block_id=f"owner_{index}",
            )
        )
        child.logger.warning(f"Child {index}")
    view = DiagnosticViewer(lambda: outer.activity)
    view.set_block(children[0])
    assert [e["message"] for e in viewer_events(view)] == ["Child 0"]
    assert len(records(children[1])) == 1
    view.close()


def test_parallel_instance_logs_aggregate_only_for_selected_logical_block(app):
    graph = parallel_parts_graph(builtin_registry())
    graph.run()
    parent = next(b for b in graph.blocks.values() if b.type_id == "flow.inspect_each")
    child = next(iter(parent.graph.blocks.values()))
    view = DiagnosticViewer(lambda: graph.activity)
    view.set_block(child)
    finishes = [e for e in viewer_events(view) if e["kind"] == "finish"]
    assert len(finishes) == 25
    assert all(e["source"]["logical_path"] == child.logger.name for e in finishes)
    assert all("/part-" in e["source"]["path"] for e in finishes)
    view.close()


def test_live_worker_logging_and_project_replacement_use_correct_shared_store(app):
    from test_editor import pump_until

    window = MainWindow()
    window.registry.register(DiagnosticBlock)
    graph = Graph()
    block = graph.add(window.registry.create(DiagnosticBlock.type_id))
    source = graph.add(window.registry.create("input.demo"))
    graph.connect(source.id, "image", block.id, "image")
    window.replace_graph(graph)
    window.inspector.select(block)
    for count in (1, 2):
        window.start_run(False)
        pump_until(app, lambda: window.runner is None)
        window.diagnostics_view.refresh()
        messages = [e for e in viewer_events(window.diagnostics_view) if e["kind"] == "diagnostic"]
        assert len(messages) == count
    window.replace_graph(Graph())
    assert not viewer_events(window.diagnostics_view)
    window.dirty = False
    window.close()
