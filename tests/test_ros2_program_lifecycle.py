"""Self-starting real-image/data ROS examples; native DDS checks are opt-in."""

import hashlib
import json
import os
import runpy
from pathlib import Path

import cv2
import numpy as np
import pytest
from test_program_lifecycle import alive, wait_for

from opencv_dataflow.assistant_api import validate_graph
from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph

BASE = (
    Path(__file__).resolve().parents[1] / "examples/industrial integration/ros2_program_lifecycle"
)


def test_managed_ros_examples_static_assets_and_regeneration(tmp_path):
    source = json.loads((BASE / "SOURCES.json").read_bytes())
    assert hashlib.sha256((BASE / "pcb.ppm").read_bytes()).hexdigest() == source["output_sha256"]
    paths = sorted([*BASE.glob("*.json"), BASE / "pcb.ppm"])
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    runpy.run_path(str(BASE / "build_example.py"), run_name="__main__")
    assert before == [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    for name in ("image", "data"):
        graph = Graph.load(BASE / f"managed_{name}.json", builtin_registry())
        try:
            assert validate_graph(graph)["valid"]
            assert not graph.program_status()["started"]
            folder = tmp_path / name
            folder.mkdir()
            graph.save(folder / "program.json")
            for asset in graph.program["assets"]:
                assert (folder / asset).read_bytes() == (BASE / asset).read_bytes()
            assert "CC BY 4.0" in graph.blocks["instructions"].description
        finally:
            graph.close()


@pytest.mark.skipif(os.environ.get("OPENCV_DATAFLOW_TEST_ROS2") != "1", reason="Opt-in native ROS")
@pytest.mark.parametrize("case", ("image", "data"))
@pytest.mark.parametrize("profile", ("debug", "lean"))
def test_first_run_starts_ros_and_close_stops_only_owned_nodes(case, profile):
    graph = Graph.load(BASE / f"managed_{case}.json", builtin_registry())
    pids = []
    try:
        graph.set_run_profile(profile)
        assert not graph.program_status()["started"]
        graph.run()
        source = graph.blocks["source"]
        assert source.state == "pass", source.error
        assert source.current_emissions
        status = graph.program_status()
        assert status["services"]["ros_demo"]["running"] and status["run_index"] == 1
        pids = [status["services"]["ros_demo"]["pid"], source._client.process.pid]
        if case == "image":
            image = source.last_result.outputs["image"]
            expected = cv2.imread(str(BASE.parent.parent / "assets/showcase/visa-pcb2-0000.png"))
            np.testing.assert_array_equal(image, cv2.cvtColor(expected, cv2.COLOR_BGR2RGB))
            assert graph.blocks["edges"].state == "pass"
        else:
            assert source.last_result.outputs["data"]["source"] == "program-owned demo"
        graph.run()
        assert graph.program_status()["services"]["ros_demo"]["pid"] == pids[0]
        assert graph.program_status()["state"]["cycles"] == 2
    finally:
        graph.close()
    wait_for(lambda: not any(alive(pid) for pid in pids))
    status = graph.program_status()
    assert status["closed"] and "ROS_DEMO_STOPPED" in status["services"]["ros_demo"]["output_tail"]


@pytest.mark.skipif(os.environ.get("OPENCV_DATAFLOW_TEST_ROS2") != "1", reason="Opt-in native ROS")
def test_application_quit_path_joins_worker_and_closes_native_ros(app):
    from test_program_lifecycle_gui import pump_until

    from opencv_dataflow.app import MainWindow

    graph = Graph.load(BASE / "managed_image.json", builtin_registry())
    window = MainWindow()
    window.replace_graph(graph)
    try:
        window.start_run(True)
        pump_until(
            app, lambda: graph.program_status()["services"].get("ros_demo", {}).get("running"), 40
        )
        pid = graph.program_status()["services"]["ros_demo"]["pid"]
        window.shutdown_application()
        pump_until(app, lambda: window.runner is None)
        wait_for(lambda: not alive(pid))
        status = graph.program_status()
        assert status["closed"]
        assert "ROS_DEMO_STOPPED" in status["services"]["ros_demo"]["output_tail"]
    finally:
        window.shutdown_application()
        window.dirty = False
        window.close()
        app.processEvents()
