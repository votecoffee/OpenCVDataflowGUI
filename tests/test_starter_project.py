import json
import subprocess
import sys

import numpy as np
import pytest

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.examples import starter_graph, template_graph
from opencv_dataflow.runner import main
from opencv_dataflow.toolblocks import encapsulate


def test_saved_starter_matches_builder_and_runs_repeatedly():
    registry = builtin_registry()
    graph = Graph.load(
        "examples/getting_started/starter_inspection/starter_inspection.json", registry
    )
    assert graph.to_dict() == starter_graph(registry).to_dict()
    previous = None
    for _ in range(3):
        graph.run()
        assert all(b.state == "pass" for b in graph.blocks.values())
        measured = graph.blocks["starter_components"].last_result
        assert measured.outputs["count"] == 2 and measured.score == 1
        centers = sorted(measured.outputs["centroids"])
        np.testing.assert_allclose(centers, [[105, 115], [297.5, 112.5]])
        assert graph.blocks["starter_edges"].last_output_images["image"].any()
        channel = graph.blocks["starter_histogram"].last_result.outputs["histogram"]["channels"][0]
        assert sum(channel["counts"]) == 400 * 220
        mask = graph.blocks["starter_morphology"].last_output_images["image"]
        if previous is not None:
            np.testing.assert_array_equal(mask, previous)
        previous = mask.copy()
    graph.close()


def test_starter_can_be_encapsulated_and_reopened(tmp_path):
    registry = builtin_registry()
    graph = starter_graph(registry)
    selected = [
        "starter_gray",
        "starter_median",
        "starter_threshold",
        "starter_morphology",
        "starter_components",
    ]
    block = encapsulate(graph, selected, "Inspect parts", registry)
    path = tmp_path / "nested.json"
    graph.save(path)
    loaded = Graph.load(path, registry)
    loaded.run()
    assert all(b.state == "pass" for b in loaded.blocks.values())
    assert (
        loaded.blocks[block.id].graph.blocks["starter_components"].last_result.outputs["count"] == 2
    )


def test_template_project_matches_builder_and_known_patch():
    registry = builtin_registry()
    graph = Graph.load(
        "examples/pattern_matching/template_matching/template_matching.json", registry
    )
    assert graph.to_dict() == template_graph(registry).to_dict()
    graph.run()
    match = graph.blocks["match_result"]
    assert match.state == "pass", match.error
    assert match.last_result.outputs["score"] > 0.999
    assert match.last_result.outputs["location"]["top_left"] == [250, 85]


def test_headless_report_repeats_and_preserves_existing_receipt(tmp_path):
    report = tmp_path / "run.json"
    assert (
        main(
            [
                "examples/getting_started/starter_inspection/starter_inspection.json",
                "--repeat",
                "3",
                "--report",
                str(report),
            ]
        )
        == 0
    )
    receipt = json.loads(report.read_text())
    assert receipt["passed"] and len(receipt["runs"]) == 3
    assert all(run["passed"] for run in receipt["runs"])
    components = next(b for b in receipt["runs"][0]["blocks"] if b["id"] == "starter_components")
    assert components["outputs"]["count"] == 2
    assert "overlay" not in components["outputs"]
    original = report.read_bytes()
    assert (
        main(
            [
                "examples/getting_started/starter_inspection/starter_inspection.json",
                "--report",
                str(report),
            ]
        )
        == 2
    )
    assert report.read_bytes() == original


def test_headless_failure_status_missing_files_and_bad_repeat(tmp_path):
    registry = builtin_registry()
    graph = starter_graph(registry)
    graph.blocks["starter_components"].parameters["expected_count"] = 3
    path = tmp_path / "fail.json"
    graph.save(path)
    report = tmp_path / "failure.json"
    assert main([str(path), "--report", str(report)]) == 1
    assert not json.loads(report.read_text())["passed"]
    graph.blocks["starter_median"].parameters["kernel"] = 4
    graph.save(path)
    assert main([str(path)]) == 1
    assert main([str(tmp_path / "missing.json")]) == 2
    with pytest.raises(SystemExit) as exc:
        main([str(path), "--repeat", "0"])
    assert exc.value.code == 2


def test_actual_headless_entry_does_not_import_qt():
    code = (
        "import sys; from opencv_dataflow.runner import main; "
        "assert not any(k.startswith('PySide6') for k in sys.modules); "
        "raise SystemExit(main(['examples/pattern_matching/"
        "template_matching/template_matching.json']))"
    )
    process = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=20
    )
    assert process.returncode == 0, process.stderr
    assert "PASS (3 blocks)" in process.stdout
