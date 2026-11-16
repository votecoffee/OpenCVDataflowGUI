"""Offscreen editor checks: program ownership survives worker copies and ends on close."""

import copy
import json
import time

from test_program_lifecycle import alive, service, wait_for

from opencv_dataflow.app import MainWindow
from opencv_dataflow.blocks import builtin_registry, demo_graph
from opencv_dataflow.core import Graph
from opencv_dataflow.program_settings import ProgramSettingsDialog


def pump_until(app, predicate, timeout=20):
    end = time.monotonic() + timeout
    while not predicate():
        app.processEvents()
        assert time.monotonic() < end
        time.sleep(0.01)


def loaded_window(tmp_path):
    graph = demo_graph(builtin_registry())
    graph.bind_project_dir(tmp_path)
    graph.set_program({"enabled": True, "services": [service(tmp_path)]})
    window = MainWindow()
    window.replace_graph(graph)
    return window, graph


def test_gui_run_once_reuses_program_service_and_new_closes_it(app, tmp_path):
    window, graph = loaded_window(tmp_path)
    try:
        assert not graph.program_status()["started"]
        window.start_run(False)
        pump_until(app, lambda: window.runner is None)
        first = graph.program_status()
        assert first["started"] and first["run_index"] == 1
        pids = json.loads((tmp_path / "pids.json").read_bytes())
        window.start_run(False)
        pump_until(app, lambda: window.runner is None)
        second = graph.program_status()
        assert second["run_index"] == 2
        assert second["services"]["demo"]["pid"] == first["services"]["demo"]["pid"]
        assert all(alive(pid) for pid in pids)
        window.replace_graph(Graph())
        assert graph.program_status()["closed"]
        wait_for(lambda: not any(alive(pid) for pid in pids))
    finally:
        window.dirty = False
        window.close()
        app.processEvents()


def test_close_during_continuous_waits_for_worker_then_shuts_program(app, tmp_path):
    window, graph = loaded_window(tmp_path)
    try:
        window.start_run(True)
        pump_until(
            app, lambda: graph.program_status()["started"] and (tmp_path / "pids.json").exists()
        )
        pids = json.loads((tmp_path / "pids.json").read_bytes())
        window.dirty = False
        window.close()
        pump_until(app, lambda: window.runner is None and window._shutdown_complete)
        wait_for(lambda: not any(alive(pid) for pid in pids))
        assert (tmp_path / "stopped.txt").read_text() == "graceful"
    finally:
        window.dirty = False
        if window.runner is not None:
            window.stop_run()
            pump_until(app, lambda: window.runner is None)
        window.close()
        app.processEvents()


def test_program_dialog_drafts_validation_services_and_runtime(app, tmp_path):
    graph = Graph()
    graph.bind_project_dir(tmp_path)
    original = copy.deepcopy(graph.program)
    dialog = ProgramSettingsDialog(graph)
    try:
        dialog.name.setText("Discarded draft")
        dialog.reject()
        assert graph.program == original
    finally:
        dialog.close()
    dialog = ProgramSettingsDialog(graph)
    try:
        dialog.name.setText("Inspection program")
        dialog.enabled.setChecked(True)
        dialog.variables.setPlainText('{"inspections": 0}')
        dialog.scripts["startup"].setPlainText("if bad")
        dialog.apply_settings()
        assert dialog.error.text() and graph.program == original
        dialog.scripts["startup"].setPlainText("state['started'] = True")
        dialog.add_service()
        dialog.service_fields["argv"].setPlainText("{python}\n-u\n{project_dir}/service.py")
        dialog.service_fields["enabled"].setChecked(False)
        dialog.apply_settings()
        assert graph.program["name"] == "Inspection program"
        assert graph.program["services"][0]["enabled"] is False
        assert not graph.program_status()["started"]
        graph.run()
        dialog.refresh_status()
        assert json.loads(dialog.runtime_text.toPlainText())["state"]["started"] is True
        dialog.stop_program()
        assert json.loads(dialog.runtime_text.toPlainText())["closed"] is True
    finally:
        graph.close()
        dialog.close()


def test_background_acquisition_does_not_duplicate_program_hooks():
    from test_image_queue import setup_background

    from opencv_dataflow.queued_execution import QueuedExecution

    registry, graph, *_ = setup_background()
    graph.set_program(
        {
            "enabled": True,
            "scripts": {
                "startup": "state['starts'] = state.get('starts', 0) + 1",
                "before_run": "state['runs'] = state.get('runs', 0) + 1",
            },
        }
    )
    try:
        with QueuedExecution(graph, registry) as execution:
            assert execution.run_one()
            assert execution.run_one()
        state = graph.program_status()["state"]
        assert state == {"starts": 1, "runs": 2}
    finally:
        graph.close()
